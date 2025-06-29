from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class HATSequenceState:
    # Store bytes of the current word and first byte of new word
    curr_word_bytes: List[int]
    new_word_first_bytes: Optional[List[int]]
    is_partial_prefill: bool
    is_prefill: bool
    num_prompt_tokens: int
    num_computed_tokens_backbone: int
    num_scheduled_tokens_backbone: int
    block_table_backbone: List[List[int]]
    word_lens_bytes: List[int]
    num_scheduled_tokens_byte: int
    len_last_word_chunked: int
    multi_bytes: int

    # Predictive word embedding from the previous word (needed by the decoder)
    prev_pred_backbone_embedding: Optional[torch.Tensor] # Shape [word_windwow_size, D]

    # Encoder outputs for the current word (needed by encoder_connector and decoder)
    encoder_embeds_curr_word: List[torch.Tensor]
    encoder_embeds_new_word: List[torch.Tensor]

    # Byte and word position of the current sequence we are generating tokens for
    word_position: torch.Tensor
    word_position_cpu: int
    byte_position: int


@dataclass
class HATKVCacheState:
    num_curr_word_bytes: int
    num_computed_tokens_backbone: int
    num_computed_tokens_byte: int


@dataclass
class HATBatchInput:
    word_positions: Optional[torch.Tensor] = None
    word_len_bytes: Optional[torch.Tensor] = None
    predictive_word_embeddings: Optional[torch.Tensor] = None
    latent_word_embeddings: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


class HATSubmodelRole(str, Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    BACKBONE = "backbone"

    
def _create_empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
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
                             finished_sending=None,
                             finished_recving=None)
    
    
def safe_tensor_slice(tensor: torch.Tensor, count: int, keep_prefix: bool=True) -> Optional[torch.Tensor]:
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
        
        
def safe_list_slice(list: List, count: int, keep_prefix: bool=True) -> Optional[List]:
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
            
            
def split_text(hat_splitter: HATRuleSplitter, text_bytes: List[int]) -> List[List[int]]:
    """Splits a text into its constituent words in bytes."""
    prev_num_bytes = len(text_bytes)
    text = hat_splitter.decode(text_bytes, skip_special_tokens=False, errors="ignore")
    list_of_words_in_bytes = hat_splitter.encode(text)
    after_num_bytes = sum(len(word) for word in list_of_words_in_bytes)
    
    diff = prev_num_bytes - after_num_bytes
    if diff > 0:
        if len(list_of_words_in_bytes) > 1:
            list_of_words_in_bytes[-1].extend(text_bytes[-diff:])
        else:
            list_of_words_in_bytes = [text_bytes]
    return list_of_words_in_bytes


def check_byte_for_new_word(hat_splitter: HATRuleSplitter, word: List[int]) -> Tuple[bool, Optional[List[List[int]]]]:
    if len(word) > hat_splitter.max_word_size:
        return True, None
    try:
        word_str = hat_splitter.decode(word, errors="strict", skip_special_tokens=False)
        w = hat_splitter.encode(word_str)
    except UnicodeDecodeError as e:
        if "invalid start byte" in e.reason:
            return True, None
        else:
            return False, None
    return len(w) > 1, w