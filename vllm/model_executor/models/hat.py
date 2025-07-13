# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.vllm_flash_attn import flash_attn_varlen_func

from .interfaces import SupportsLoRA
from .utils import AutoWeightsLoader, is_pp_missing_parameter, maybe_prefix


class HATAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_scaling["is_neox_style"]
            if "is_neox_style" in rope_scaling else True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class HATCrossAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 hidden_size: int,
                 q_hidden_size: int,
                 kv_hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.q_hidden_size = q_hidden_size
        self.kv_hidden_size = kv_hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.kv_size = self.num_kv_heads * self.head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = ColumnParallelLinear(
            input_size=self.q_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        self.kv_proj = QKVParallelLinear(
            hidden_size=self.kv_hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.q_hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_scaling["is_neox_style"]
            if "is_neox_style" in rope_scaling else True,
        )

    def forward(
        self,
        q_position_ids: torch.Tensor,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        kv_position_ids: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        force_attn: bool = False,
        scheduler_metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(q_input)
        kv, _ = self.kv_proj(kv_input)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)

        # L TODO: Fix this later
        q, _ = self.rotary_emb(q_position_ids, q, torch.zeros_like(q))
        _, k = self.rotary_emb(kv_position_ids, torch.zeros_like(k), k)

        shape = q.shape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty(shape, dtype=q.dtype, device=q.device)
        output = output.view(-1, self.num_heads, self.head_dim)
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=2,
            scheduler_metadata=scheduler_metadata,
            causal=False,
        )

        output, _ = self.o_proj(output.view(shape))

        return output


class HATGuideVectorAdd(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 hidden_size: int,
                 q_hidden_size: int,
                 kv_hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.q_hidden_size = q_hidden_size
        self.kv_hidden_size = kv_hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(
            input_size=self.q_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        self.k_proj = ColumnParallelLinear(
            input_size=self.kv_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )

        self.v_proj = ColumnParallelLinear(
            input_size=self.kv_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.q_hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        word_embeddings: torch.Tensor,
        word_lens_bytes: torch.Tensor,
    ) -> torch.Tensor:
        v, _ = self.v_proj(word_embeddings)

        output, _ = self.o_proj(v)
        if word_lens_bytes is not None:
            output = output.repeat_interleave(word_lens_bytes, dim=0)
        return output


class HATMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class HATTransformerLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias

        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.self_attn = HATAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = HATMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HATDecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_type: type[nn.Module] = HATTransformerLayer,
    ) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias

        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.query_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(config.cross_attention_config.hidden_size_kv,
                               eps=config.rms_norm_eps)

        self.cross_attention = HATGuideVectorAdd(
            config=config,
            hidden_size=config.cross_attention_config.hidden_size,
            q_hidden_size=config.cross_attention_config.hidden_size_q,
            kv_hidden_size=config.cross_attention_config.hidden_size_kv,
            num_heads=config.cross_attention_config.num_attention_heads,
            num_kv_heads=getattr(
                config.cross_attention_config, "attention_num_kv_heads",
                config.cross_attention_config.num_attention_heads),
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.cross_attn",
        )

        self.llama_layer = layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.llama_layer")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        word_lens_bytes: torch.Tensor,
        predictive_word_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.query_norm(hidden_states)
        else:
            hidden_states, residual = self.query_norm(hidden_states, residual)
        word_embeddings = self.kv_norm(predictive_word_embeddings)

        hidden_states = self.cross_attention(
            word_embeddings=word_embeddings,
            word_lens_bytes=word_lens_bytes,
        )

        hidden_states, residual = self.llama_layer(positions, hidden_states,
                                                   residual)
        return hidden_states, residual


class HATEncoderConnector(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.cross_attention_config.hidden_size
        self.quant_config = quant_config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias

        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.latent_query = torch.nn.Parameter(
            torch.empty(1, 1, self.hidden_size))

        self.cross_attention_encoder_connector = HATCrossAttention(
            config=config,
            hidden_size=self.hidden_size,
            q_hidden_size=config.cross_attention_config.hidden_size_q,
            kv_hidden_size=config.cross_attention_config.hidden_size_kv,
            num_heads=config.cross_attention_config.num_attention_heads,
            num_kv_heads=getattr(
                config.cross_attention_config, "attention_num_kv_heads",
                config.cross_attention_config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.cross_attn",
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        byte_positions: torch.Tensor,
        word_positions: torch.Tensor,
        word_lens_bytes_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # L TODO: Fix this
        latent = self.latent_query.squeeze(0)
        # L TODO: Remove expand
        latent_word_embeddings = latent.expand(word_positions.shape[0], -1)

        cu_seqlens_q = torch.arange(word_positions.shape[0] + 1,
                                    device=word_positions.device,
                                    dtype=torch.int32)
        cu_seqlens_k = torch.cumsum(word_lens_bytes_flat,
                                    dim=0,
                                    dtype=torch.int32)
        max_seqlen_k = word_lens_bytes_flat.max()

        #scheduler_metadata = get_scheduler_metadata(
        #        batch_size=word_positions.shape[0],
        #        max_seqlen_q=1,
        #        max_seqlen_k=max_seqlen_k,
        #        cache_seqlens=torch.zeros(0, device=word_positions.device, dtype=torch.int32),
        #        num_heads_q=self.config.cross_attention_config.num_attention_heads,
        #        num_heads_kv=getattr(self.config.cross_attention_config, "attention_num_kv_heads",
        #                            self.config.cross_attention_config.num_attention_heads),
        #        headdim=self.config.head_dim,
        #        page_size=256,
        #        cu_seqlens_q=cu_seqlens_q,
        #        causal=False,
        #    )

        updated_latent_word_embeddings = self.cross_attention_encoder_connector(
            q_position_ids=word_positions,
            q_input=latent_word_embeddings,
            kv_input=encoder_hidden_states,
            kv_position_ids=byte_positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            force_attn=True,
            scheduler_metadata=None,
        )
        return updated_latent_word_embeddings

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        return _load_weights(self, weights)


@support_torch_compile
class HATEncoderModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATTransformerLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embedding_layer = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(
                layer_type(config=config,
                           cache_config=cache_config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.layers.{i}"))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is not None:
            hidden_states = hidden_states + residual

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        return _load_weights(self, weights)


class HATEncoderForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATTransformerLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.encoder = self._init_model(vllm_config=vllm_config,
                                        prefix=maybe_prefix(prefix, "encoder"),
                                        layer_type=layer_type)

        self.encoder_connector = HATEncoderConnector(
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="encoder_connector")

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = HATTransformerLayer):
        return HATEncoderModel(vllm_config=vllm_config,
                               prefix=prefix,
                               layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_output = self.encoder(input_ids, positions, inputs_embeds)
        return model_output

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self,
                                   skip_prefixes=[
                                       "backbone", "decoder_connector",
                                       "lm_head", "layer_norm", "decoder"
                                   ])
        return loader.load_weights(
            (name, loaded_weight) for name, loaded_weight in weights)


@support_torch_compile
class HATDecoderModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.decoder_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.decoder_layers.append(
                layer_type(vllm_config=vllm_config,
                           cache_config=cache_config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.decoder_layers.{i}"))

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        word_lens_bytes: torch.Tensor,
        predictive_word_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = previous_hidden_states
        residual = None

        for layer in self.decoder_layers:
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            word_lens_bytes,
                                            predictive_word_embeddings)

        if residual is not None:
            hidden_states = hidden_states + residual

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        return _load_weights(self, weights)


class HATDecoderForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "self_attn.qkv_proj":
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "cross_attn.kv_proj": ["cross_attn.k_proj", "cross_attn.v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATDecoderLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.decoder = self._init_model(vllm_config=vllm_config,
                                        prefix=maybe_prefix(prefix, "decoder"),
                                        layer_type=layer_type)

        self.layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)

        self.sampler = get_sampler()

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = HATDecoderLayer):
        return HATDecoderModel(vllm_config=vllm_config,
                               prefix=prefix,
                               layer_type=layer_type)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: Optional[torch.Tensor] = None,
        word_lens_bytes: Optional[torch.Tensor] = None,
        predictive_word_embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:

        model_output = self.decoder(positions, previous_hidden_states,
                                    word_lens_bytes,
                                    predictive_word_embeddings)
        model_output = self.layer_norm(model_output)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self,
                                   skip_prefixes=[
                                       "encoder", "backbone",
                                       "decoder_connector", "encoder_connector"
                                   ])
        return loader.load_weights(
            (name, loaded_weight) for name, loaded_weight in weights)


@support_torch_compile
class HATBackboneModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATTransformerLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(
                layer_type(config=config,
                           cache_config=cache_config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.layers.{i}"))

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = previous_hidden_states
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is not None:
            hidden_states = hidden_states + residual

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        return _load_weights(self, weights)


class HATBackboneForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "self_attn.qkv_proj":
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "cross_attention_encoder_connector.kv_proj": [
            "cross_attention_encoder_connector.k_proj",
            "cross_attention_encoder_connector.v_proj"
        ],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = HATTransformerLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.backbone = self._init_model(vllm_config=vllm_config,
                                         prefix=maybe_prefix(
                                             prefix, "backbone"),
                                         layer_type=layer_type)

        # Hack to satisfy the weight loader
        self.decoder_connector = nn.Module()
        first_word_embedding = torch.nn.Parameter(
            torch.empty(1, 1, config.hidden_size))
        self.decoder_connector.register_parameter("first_word_embedding",
                                                  first_word_embedding)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = HATTransformerLayer):
        return HATBackboneModel(vllm_config=vllm_config,
                                prefix=prefix,
                                layer_type=layer_type)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        model_output = self.backbone(positions, previous_hidden_states)
        return model_output

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=[
                                       "encoder", "decoder", "lm_head",
                                       "layer_norm", "encoder_connector"
                                   ])
        return loader.load_weights(
            (name, loaded_weight) for name, loaded_weight in weights)


def _load_weights(self, weights: Iterable[Tuple[str,
                                                torch.Tensor]]) -> Set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
        ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
        ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ("cross_attention_encoder_connector.kv_proj",
         "cross_attention_encoder_connector.k_proj", "k"),
        ("cross_attention_encoder_connector.kv_proj",
         "cross_attention_encoder_connector.v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_params: Set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        if ("rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name):
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        if (self.quant_config is not None
                and (scale_name := self.quant_config.get_cache_scale(name))):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            loaded_weight = (loaded_weight
                             if loaded_weight.dim() == 0 else loaded_weight[0])
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        if "scale" in name:
            # Remapping the name of FP8 kv-scale.
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)

            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)

    return loaded_params
