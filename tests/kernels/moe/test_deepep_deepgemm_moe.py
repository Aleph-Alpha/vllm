# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepEP + DeepGEMM integration
DeepGEMM are gemm kernels specialized for the
fp8 block-quantized case.
"""

import dataclasses
from typing import Optional

import pytest
import torch.distributed
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.platforms import current_platform
from vllm.utils import has_deep_ep, has_deep_gemm
from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used

from .parallel_utils import ProcessGroupInfo, parallel_launch
from .utils import make_test_weights

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (  # noqa: E501
        DeepEPHTPrepareAndFinalize)
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
        DeepEPLLPrepareAndFinalize)

    from .parallel_utils import DeepEPHTArgs, DeepEPLLArgs, make_deepep_a2a

if has_deep_gemm():

    from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts)
    from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
        DeepGemmExperts)

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(),
    reason="Requires deep_ep kernels",
)

requires_deep_gemm = pytest.mark.skipif(
    not has_deep_gemm(),
    reason="Requires deep_gemm kernels",
)

P = ParamSpec("P")


def next_power_of_2(x):
    import math
    if x == 0:
        return 1
    return 2**math.ceil(math.log2(x))


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1q, w2q, w1_scale, w2_scale
    """
    w1, w1q, w1_scale, w2, w2q, w2_scale = make_test_weights(
        e, n, k, torch.bfloat16, torch.float8_e4m3fn, block_size)
    return w1q, w2q, w1_scale, w2_scale


@dataclasses.dataclass
class TestConfig:
    topk: int
    m: int
    k: int
    n: int
    num_experts: int
    per_act_token_quant: bool
    block_size: list[int]
    # configs for testing low-latency kernels
    low_latency: bool
    use_fp8_dispatch: Optional[bool] = False


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor  # all ranks make this many tokens
    rank_token_scales: Optional[torch.Tensor]
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig, rank) -> "TestTensors":

        dtype = torch.bfloat16
        topk, m, k = (config.topk, config.m, config.k)

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        rank_tokens = torch.randn(
            (m, k), device=torch.cuda.current_device(), dtype=dtype) / 10.0
        rank_tokens = rank_tokens.clamp(min=fp8_min, max=fp8_max)
        rank_token_scales = None

        topk_ids = torch.randint(
            low=0,
            high=config.num_experts,
            size=(m, topk),
            device=torch.cuda.current_device()).to(dtype=torch.int64)

        topk_weights = torch.randn(topk_ids.shape,
                                   dtype=torch.float32,
                                   device=torch.cuda.current_device())

        return TestTensors(rank_tokens=rank_tokens,
                           rank_token_scales=rank_token_scales,
                           topk=topk_ids,
                           topk_weights=topk_weights,
                           config=config)


def make_ll_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo,
                           max_tokens_per_rank: int, dp_size: int,
                           hidden_size: int, q_dtype: Optional[torch.dtype],
                           test_config: TestConfig) -> FusedMoEModularKernel:

    assert test_config.low_latency
    assert test_config.use_fp8_dispatch is not None

    a2a: DeepEPLLPrepareAndFinalize = make_deepep_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        deepep_ht_args=None,
        deepep_ll_args=DeepEPLLArgs(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            num_experts=test_config.num_experts,
            use_fp8_dispatch=test_config.use_fp8_dispatch),
        q_dtype=q_dtype,
        block_shape=test_config.block_size)

    fused_experts = BatchedDeepGemmExperts(
        max_num_tokens=max_tokens_per_rank,
        num_dispatchers=pgi.world_size // dp_size,
        block_shape=test_config.block_size,
        per_act_token_quant=test_config.per_act_token_quant)
    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def make_ht_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo,
                           dp_size: int, num_local_experts: int,
                           q_dtype: Optional[torch.dtype],
                           test_config: TestConfig) -> FusedMoEModularKernel:

    assert not test_config.low_latency
    assert test_config.use_fp8_dispatch is None

    a2a: DeepEPHTPrepareAndFinalize = make_deepep_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        deepep_ht_args=DeepEPHTArgs(num_local_experts=num_local_experts),
        deepep_ll_args=None,
        q_dtype=q_dtype,
        block_shape=test_config.block_size)

    fused_experts = DeepGemmExperts()
    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def make_modular_kernel(pg: ProcessGroup, pgi: ProcessGroupInfo, dp_size: int,
                        num_local_experts: int,
                        test_tensors: TestTensors) -> FusedMoEModularKernel:

    q_dtype = torch.float8_e4m3fn
    test_config = test_tensors.config

    mk: FusedMoEModularKernel
    # Make modular kernel
    if test_config.low_latency:
        max_tokens_per_rank = max(
            64, next_power_of_2(test_tensors.rank_tokens.size(0)))
        hidden_size = test_tensors.rank_tokens.size(-1)

        mk = make_ll_modular_kernel(pg=pg,
                                    pgi=pgi,
                                    max_tokens_per_rank=max_tokens_per_rank,
                                    dp_size=dp_size,
                                    hidden_size=hidden_size,
                                    q_dtype=q_dtype,
                                    test_config=test_config)
    else:
        mk = make_ht_modular_kernel(pg, pgi, dp_size, num_local_experts,
                                    q_dtype, test_config)

    return mk


def deepep_deepgemm_moe_impl(pg: ProcessGroup, pgi: ProcessGroupInfo,
                             dp_size: int, test_tensors: TestTensors,
                             w1: torch.Tensor, w2: torch.Tensor,
                             w1_scale: Optional[torch.Tensor],
                             w2_scale: Optional[torch.Tensor]) -> torch.Tensor:

    test_config = test_tensors.config
    num_experts = test_config.num_experts
    num_local_experts = w1.size(0)

    def build_expert_map():
        num_local_experts = w1.size(0)
        expert_map = torch.full((num_experts, ),
                                fill_value=-1,
                                dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(),
                             dtype=torch.int32)

    # Make modular kernel
    mk: FusedMoEModularKernel = make_modular_kernel(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        num_local_experts=num_local_experts,
        test_tensors=test_tensors)

    # Low-Latency kernels can't dispatch scales.
    a1_scale = (None
                if test_config.low_latency else test_tensors.rank_token_scales)

    out = mk.forward(hidden_states=test_tensors.rank_tokens,
                     w1=w1,
                     w2=w2,
                     topk_weights=test_tensors.topk_weights,
                     topk_ids=test_tensors.topk,
                     inplace=False,
                     activation="silu",
                     global_num_experts=num_experts,
                     expert_map=build_expert_map(),
                     w1_scale=w1_scale,
                     w2_scale=w2_scale,
                     w1_zp=None,
                     w2_zp=None,
                     a1_scale=a1_scale,
                     a2_scale=None,
                     apply_router_weight_on_input=False)
    return out


def triton_impl(a: torch.Tensor, topk_ids: torch.Tensor,
                topk_weights: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                w1_scale: torch.Tensor, w2_scale: torch.Tensor,
                a1_scale: torch.Tensor, block_shape: list[int]):

    return fused_experts(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        block_shape=block_shape,
        # Make sure this is set to False so we
        # dont end up comparing the same implementation.
        allow_deep_gemm=False)


def _test_deepep_deepgemm_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    current_platform.seed_everything(pgi.rank)

    w1 = w1.to(device=torch.cuda.current_device())
    w2 = w2.to(device=torch.cuda.current_device())
    w1_scale = w1_scale.to(device=torch.cuda.current_device())
    w2_scale = w2_scale.to(device=torch.cuda.current_device())

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config, pgi.rank)
    block_shape = [
        w1.size(1) // w1_scale.size(1),
        w1.size(2) // w1_scale.size(2)
    ]

    with set_current_vllm_config(VllmConfig()):
        # Reference
        triton_moe = triton_impl(a=test_tensors.rank_tokens,
                                 topk_ids=test_tensors.topk,
                                 topk_weights=test_tensors.topk_weights,
                                 w1=w1,
                                 w2=w2,
                                 w1_scale=w1_scale,
                                 w2_scale=w2_scale,
                                 a1_scale=test_tensors.rank_token_scales,
                                 block_shape=block_shape)

        # Slice experts for this rank.
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]
        w1_scale_ep = w1_scale[e_start:e_end]
        w2_scale_ep = w2_scale[e_start:e_end]

        deepep_moe = deepep_deepgemm_moe_impl(
            pg,
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
        )

    torch.testing.assert_close(
        triton_moe,
        deepep_moe,
        atol=6e-2,
        rtol=6e-2,
    )


MNKs = [
    (8, 128, 128),
    (8, 128, 512),
    (8, 512, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (129, 128, 256),
    (129, 1024, 2048),
    (222, 1024, 2048),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
@requires_deep_gemm
@pytest.mark.skipif(is_blackwell_deep_gemm_used(),
                    reason="Skipping test for Blackwell DeepGEMM")
def test_ht_deepep_deepgemm_moe(mnk: tuple[int, int, int], num_experts: int,
                                topk: int, world_dp_size: tuple[int, int]):
    """
    Tests for High-Throughput DeepEP + DeepGemm integration.
    """
    import deep_gemm

    m, n, k = mnk
    current_platform.seed_everything(7)

    if topk > num_experts:
        pytest.skip(f"Skipping test: topk={topk} > E={num_experts}")

    block_m = deep_gemm.get_m_alignment_for_contiguous_layout()
    block_size = [block_m, block_m]

    world_size, dp_size = world_dp_size
    config = TestConfig(topk=topk,
                        m=m,
                        k=k,
                        n=n,
                        num_experts=num_experts,
                        per_act_token_quant=False,
                        block_size=block_size,
                        low_latency=False,
                        use_fp8_dispatch=None)

    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
        num_experts, n, k, block_size)

    parallel_launch(world_size, _test_deepep_deepgemm_moe, dp_size, config, w1,
                    w2, w1_scale, w2_scale)


MNKs = [
    (1, 128, 2560),
    (2, 128, 2560),
    (3, 1024, 2560),
    (32, 128, 2560),
    (45, 512, 2560),
    (64, 1024, 2560),
    (222, 1024, 2560),
]
# Fix tests for USE_FP8_DISPATCH=True
USE_FP8_DISPATCH = [False]


@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("use_fp8_dispatch", USE_FP8_DISPATCH)
@pytest.mark.parametrize("block_size", [[128, 128]])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@requires_deep_ep
@requires_deep_gemm
@pytest.mark.skipif(is_blackwell_deep_gemm_used(),
                    reason="Skipping test for Blackwell DeepGEMM")
def test_ll_deepep_deepgemm_moe(
    mnk: tuple[int, int, int],
    num_experts: int,
    topk: int,
    use_fp8_dispatch: bool,
    block_size: list[int],
    world_dp_size: tuple[int, int],
):
    """
    Tests for Low-Latency DeepEP + DeepGemm integration.
    """

    m, n, k = mnk
    current_platform.seed_everything(7)

    if topk > num_experts:
        pytest.skip(f"Skipping test: topk={topk} > E={num_experts}")

    world_size, dp_size = world_dp_size
    config = TestConfig(
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        per_act_token_quant=False,
        block_size=block_size,
        low_latency=True,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(
        num_experts, n, k, block_size)

    parallel_launch(world_size, _test_deepep_deepgemm_moe, dp_size, config, w1,
                    w2, w1_scale, w2_scale)
