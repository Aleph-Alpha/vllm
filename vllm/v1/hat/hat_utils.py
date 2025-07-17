# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput

# Constants

COMPRESSION_RATIO = 4
BYTES_PER_WORKER_STEP = 8
LIMIT_FOR_STATIC_STEPS = 8

# Data Classes


@dataclass
class HATSequenceState:
    # Store bytes of the current word and first byte of new word
    curr_word_bytes: List[int]
    new_word_first_bytes: Optional[List[int]]
    all_token_ids: List[int]

    # These are updated for each step
    num_scheduled_tokens_backbone: int
    num_scheduled_tokens_byte: int
    word_lens_bytes: List[int]
    block_table_backbone: List[List[int]]

    # Chunked prefill specific
    len_last_word_chunked: int
    multi_bytes: int

    # Predictive word embedding from the previous word (needed by the decoder)
    prev_pred_backbone_embedding: Optional[
        torch.Tensor]  # Shape [word_windwow_size, hidden_size]

    # Encoder outputs for the current word (needed by encoder_connector and decoder)
    encoder_embeds_curr_word: List[torch.Tensor]
    encoder_embeds_new_word: List[torch.Tensor]

    # Byte and word position of the current sequence we are generating tokens for
    word_position_cpu: torch.Tensor
    byte_position: int

    request_type: "HATRequestType"
    
    # Edge case: A sequence that resumes from preemption with a small chunked prefill 
    # (less than a full word). 
    # The scheduler sets `resumed_from_preemption` only on the  very first resumption 
    # of the sequence. If the first received chunk after resumption is less than a full 
    # word, the backbone ModelRunner isn't called, so it won't know that the sequence 
    # was resumed from preemption.
    # This flag identifies such scenarios.
    is_small_chunked_prefill_after_preemption: bool


@dataclass
class HATKVCacheState:
    num_curr_word_bytes: int
    num_computed_tokens_backbone: int
    num_computed_tokens_byte: int


@dataclass
class HATBatchInput:
    word_lens_bytes: Optional[torch.Tensor] = None
    predictive_word_embeddings: Optional[torch.Tensor] = None
    latent_word_embeddings: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class HATEncoderHiddenStatesPhases:
    encoder_hidden_states_encoder_connector: List[torch.Tensor]
    encoder_hidden_states_enc_dec_loop: Optional[torch.Tensor]
    encoder_hidden_states_final_decoder: Optional[torch.Tensor]


@dataclass
class HATEncoderConnectorInput:
    encoder_hidden_states: torch.Tensor
    byte_positions: torch.Tensor
    word_positions: torch.Tensor
    word_lens_bytes_flat: torch.Tensor


# Enums


class HATSubmodelRole(str, Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    BACKBONE = "backbone"


class HATRequestType(str, Enum):
    PREFILL = "prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    DECODE = "decode"
    DECODE_WORD_BOUNDARY = "decode_word_boundary"


# Helper Functions


def _create_empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
        kv_connector_metadata=None,
    )


def _create_empty_model_runner_output() -> ModelRunnerOutput:
    return ModelRunnerOutput(req_ids=[],
                             req_id_to_index={},
                             sampled_token_ids=[],
                             spec_token_ids=None,
                             logprobs=None,
                             prompt_logprobs_dict={},
                             pooler_output=[],
                             finished_sending=None,
                             finished_recving=None)


def safe_tensor_slice(tensor: torch.Tensor,
                      count: int,
                      keep_prefix: bool = True) -> Optional[torch.Tensor]:
    if count == 0:
        if keep_prefix:
            return tensor
        else:
            return torch.empty(0, tensor.shape[1], device=tensor.device)
    else:
        if keep_prefix:
            return tensor[:-count, :]
        else:
            return tensor[-count:, :]


def safe_list_slice(list: List,
                    count: int,
                    keep_prefix: bool = True) -> Optional[List]:
    if count == 0:
        if keep_prefix:
            return list
        else:
            return []
    else:
        if keep_prefix:
            return list[:-count]
        else:
            return list[-count:]


def split_text(hat_splitter: HATRuleSplitter,
               text_bytes: List[int]) -> List[List[int]]:
    """Splits a text into its constituent words in bytes."""
    prev_num_bytes = len(text_bytes)
    text = hat_splitter.decode(text_bytes,
                               skip_special_tokens=False,
                               errors="ignore")
    list_of_words_in_bytes = hat_splitter.encode(text)
    after_num_bytes = sum(len(word) for word in list_of_words_in_bytes)

    # For incomplete multi-byte characters, we need to add the remaining bytes to the last word
    # This can happen for chunked prefills
    diff = prev_num_bytes - after_num_bytes
    if diff > 0:
        if len(list_of_words_in_bytes) > 1:
            list_of_words_in_bytes[-1].extend(text_bytes[-diff:])
        else:
            list_of_words_in_bytes = [text_bytes]
    return list_of_words_in_bytes


def check_byte_for_new_word(
        hat_splitter: HATRuleSplitter,
        word: List[int]) -> Tuple[bool, Optional[List[List[int]]]]:
    if len(word) > hat_splitter.max_word_size:
        return True, None
    try:
        word_str = hat_splitter.decode(word,
                                       errors="strict",
                                       skip_special_tokens=False)
        w = hat_splitter.encode(word_str)
    except UnicodeDecodeError as e:
        if "invalid start byte" in e.reason:
            return True, None
        else:
            return False, None
    return len(w) > 1, w


def _allocate_kv_cache_tensors(
        kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """
    Initializes the KV cache buffer with the correct size. The buffer needs
    to be reshaped to the desired shape before being used by the models.

    Args:
        kv_cache_config: The KV cache config 
    Returns:
        dict[str, torch.Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
        """
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size,
                             dtype=torch.int8,
                             device="cuda")
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        layer_names.update(group.layer_names)
    assert layer_names == set(kv_cache_raw_tensors.keys()
                              ), "Some layers are not correctly initialized"
    return kv_cache_raw_tensors


def _reshape_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: List[AttentionBackend],
) -> dict[str, torch.Tensor]:
    """
    Reshape the KV cache tensors to the desired shape and dtype.

    Args:
        kv_cache_config: The KV cache config 
        kv_cache_raw_tensors: The KV cache buffer of each layer, with 
        correct size but uninitialized shape.
    Returns:
        Dict[str, torch.Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    kv_caches: dict[str, torch.Tensor] = {}
    for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        for layer_name in kv_cache_group_spec.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = (raw_tensor.numel() // kv_cache_spec.page_size_bytes)
            if isinstance(kv_cache_spec, AttentionSpec):
                kv_cache_shape = attn_backends[i].get_kv_cache_shape(
                    num_blocks, kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                dtype = kv_cache_spec.dtype
                try:
                    kv_cache_stride_order = attn_backends[
                        i].get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                # The allocation respects the backend-defined stride order
                # to ensure the semantic remains consistent for each
                # backend. We first obtain the generic kv cache shape and
                # then permute it according to the stride order which could
                # result in a non-contiguous tensor.
                kv_cache_shape = tuple(kv_cache_shape[i]
                                       for i in kv_cache_stride_order)
                # Maintain original KV shape view.
                inv_order = [
                    kv_cache_stride_order.index(i)
                    for i in range(len(kv_cache_stride_order))
                ]
                kv_caches[layer_name] = kv_cache_raw_tensors[layer_name].view(
                    dtype).view(kv_cache_shape).permute(*inv_order)
            else:
                raise NotImplementedError
    return kv_caches
