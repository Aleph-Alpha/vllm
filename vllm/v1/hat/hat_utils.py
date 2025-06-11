from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import torch

from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class HATSequenceState:
    # Store bytes of the current word and first byte of new word
    curr_word_bytes: List[int]
    new_word_first_bytes: Optional[List[int]]
    is_partial_prefill: bool
    num_prompt_tokens: int
    num_computed_tokens_backbone: int
    block_table_backbone: List[List[int]]
    word_lens_bytes: List[int]

    # Predictive word embedding from the previous word (needed by the decoder)
    prev_pred_backbone_embedding: Optional[torch.Tensor] # Shape [word_windwow_size, D]

    # Encoder outputs for the current word (needed by encoder_connector and decoder)
    encoder_embeds_curr_word: List[torch.Tensor]
    encoder_embeds_new_word: List[torch.Tensor]

    # Byte and word position of the current sequence we are generating tokens for
    word_position: torch.Tensor
    byte_position: int


@dataclass
class HATBatchInfo:
    word_positions: Optional[torch.Tensor] = None
    cu_seqlen_words: Optional[torch.Tensor] = None
    max_seqlen_words: Optional[int] = None
    predictive_word_embeddings: Optional[torch.Tensor] = None
    latent_word_embeddings: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class HATOutputState:
    hidden_states: torch.Tensor
    req_id_to_index: dict[str, int]


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