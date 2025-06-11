from typing import Dict, List, Optional, Tuple
import torch
from vllm.sequence import ExecuteModelRequest
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.hat.hat_utils import HATOutputState, HATSequenceState, _create_empty_scheduler_output
from vllm.v1.outputs import SamplerOutput


class HATManager:

    def __init__(self, special_token_dict: Dict[str, int], max_word_size: int, device: torch.device, rank: int, driver_rank: int):
        self.req_ids_to_hat_state: Dict[str, HATSequenceState] = {}
        self.hat_splitter = HATRuleSplitter(special_token_dict, max_word_size=max_word_size)
        self.device = device
        self.rank = rank
        self.driver_rank = driver_rank
        
        self.is_prefill = False
        self.enc_dec_new_reqs = []
        self.enc_dec_cached_reqs = []
        self.backbone_new_reqs = []
        self.backbone_cached_reqs = []
        self.outputs: List[SamplerOutput] = []

        self.first_word_embedding: torch.Tensor = None
        
    def reset_manager(self):
        self.scheduler_output_list_enc_dec = []
        self.scheduler_output_list_backbone = []
        self.outputs = []
        self.is_prefill = False
    
    def add_request(self, scheduler_output: SchedulerOutput) -> Tuple[SchedulerOutput, SchedulerOutput]:

        scheduler_output_byte = _create_empty_scheduler_output()
        scheduler_output_word = _create_empty_scheduler_output()
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            assert req_id not in self.req_ids_to_hat_state, f"Request {req_id} already exists in HATManager"

            self._add_new_sequence(new_req_data, scheduler_output.num_scheduled_tokens[req_id])
            word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes

            # We need to split the block table into two for the encoder/decoder and the backbone
            block_table_enc_dec = new_req_data.block_ids[-1:]
            block_table_backbone = new_req_data.block_ids[:-1]
            new_req_data.block_ids = block_table_enc_dec

            scheduler_output_byte.scheduled_new_reqs.append(new_req_data)
            scheduler_output_byte.num_scheduled_tokens[req_id] = scheduler_output.num_scheduled_tokens[req_id]
            scheduler_output_byte.total_num_scheduled_tokens += scheduler_output.num_scheduled_tokens[req_id]
            scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids

            new_req_data_backbone = NewRequestData(
                req_id=req_id,
                prompt_token_ids=[0] * (len(word_lens_bytes) - 1),
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=new_req_data.sampling_params,
                block_ids=block_table_backbone,
                num_computed_tokens=0,
                lora_request=new_req_data.lora_request,
            )
            cu_word_lens_bytes = torch.cumsum(torch.tensor(word_lens_bytes), dim=0)
            num_scheduled_tokens_backbone = int(torch.searchsorted(cu_word_lens_bytes,
                                                                  scheduler_output.num_scheduled_tokens[req_id],
                                                                  right=False).item())
            scheduler_output_word.scheduled_new_reqs.append(new_req_data_backbone)
            scheduler_output_word.num_scheduled_tokens[req_id] = num_scheduled_tokens_backbone
            scheduler_output_word.total_num_scheduled_tokens += num_scheduled_tokens_backbone
            scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids
        
        #print(scheduler_output_byte)
        #print(scheduler_output_word)

        for cached_req_data in scheduler_output.scheduled_cached_reqs:
            req_id = cached_req_data.req_id
            assert req_id in self.req_ids_to_hat_state, f"Request {req_id} not found in HATManager"
            req_state = self.req_ids_to_hat_state[req_id]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            block_table_enc_dec = cached_req_data.new_block_ids[-1:]
            block_table_backbone = cached_req_data.new_block_ids[:-1]
            req_state.block_table_backbone = block_table_backbone
            cached_req_data.new_block_ids = block_table_enc_dec

            scheduler_output_byte.scheduled_cached_reqs.append(cached_req_data)
            scheduler_output_byte.num_scheduled_tokens[req_id] = num_scheduled_tokens
            scheduler_output_byte.total_num_scheduled_tokens += num_scheduled_tokens
            scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids

            # L TODO: This will be set once for NewRequestData to True (if it is partial prefill) and then at some point
            # changed to False in process_outputs when we processed the last partial prefill of that request
            if req_state.is_partial_prefill:
                self._update_sequence(cached_req_data)
                word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes

                num_scheduled_tokens_backbone = len(word_lens_bytes) - 1
                cached_req_data_backbone = CachedRequestData(
                    req_id=req_id,
                    resumed_from_preemption=cached_req_data.resumed_from_preemption,
                    new_token_ids=[0] * num_scheduled_tokens_backbone,
                    new_block_ids=block_table_backbone,
                    num_computed_tokens=self.req_ids_to_hat_state[req_id].num_computed_tokens_backbone,
                )

                scheduler_output_word.scheduled_cached_reqs.append(cached_req_data_backbone)
                scheduler_output_word.num_scheduled_tokens[req_id] = num_scheduled_tokens_backbone
                scheduler_output_word.total_num_scheduled_tokens += num_scheduled_tokens_backbone
                scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids
            else:
                self.req_ids_to_hat_state[req_id].word_lens_bytes = [1]
                

        return scheduler_output_byte, scheduler_output_word
    
    def handle_encoder_output(self, scheduler_output_byte: SchedulerOutput, 
                              scheduler_output_word: SchedulerOutput, 
                              encoder_output: HATOutputState) -> Tuple[SchedulerOutput, SchedulerOutput]:
        encoder_hidden_states = encoder_output.hidden_states
        req_id_to_index = encoder_output.req_id_to_index

        req_ids = sorted(list(req_id_to_index.keys()), key=lambda x: req_id_to_index[x])
        word_lens_bytes_per_task = [self.req_ids_to_hat_state[req_id].word_lens_bytes for req_id in req_ids]
        encoder_hidden_states_encoder_connector =[]
        encoder_hidden_states_enc_dec_loop = []

        offset = 0
        offset_beginning = 0
        for idx, word_lens_bytes in enumerate(word_lens_bytes_per_task):
            req_id = req_ids[idx]
            req_state = self.req_ids_to_hat_state[req_id]
            if req_state.is_partial_prefill:
                # We remove the overcounting in word_len_bytes due to the last computed work in the previous chunked prefill 
                num_bytes_excl_last_word = sum(word_lens_bytes[:-1]) - len(self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word)
                encoder_hidden_states_encoder_connector.append(self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.clone())
                encoder_hidden_states_encoder_connector.append(
                    encoder_hidden_states[offset_beginning:offset_beginning+num_bytes_excl_last_word, :]
                )
                offset += num_bytes_excl_last_word

                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word = [
                    encoder_hidden_states[offset:offset+word_lens_bytes[-1], :].clone()
                ]
                offset += word_lens_bytes[-1]
                offset_beginning += num_bytes_excl_last_word + word_lens_bytes[-1]
            elif len(word_lens_bytes)== 1 and word_lens_bytes[0] == 1:
                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states[offset, :].unsqueeze(0).clone()
                )
                encoder_hidden_states_enc_dec_loop.append(encoder_hidden_states[offset, :])
                offset += 1
                offset_beginning += 1
            else:
                num_bytes_excl_last_word = sum(word_lens_bytes[:-1])
                offset += num_bytes_excl_last_word
                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states[offset:offset+word_lens_bytes[-1], :].clone()
                )

                encoder_hidden_states_encoder_connector.append(
                    encoder_hidden_states[offset_beginning:offset_beginning+num_bytes_excl_last_word, :]
                )
                offset += word_lens_bytes[-1]
                offset_beginning += num_bytes_excl_last_word + word_lens_bytes[-1]
        
        encoder_hidden_states_enc_dec_loop = torch.cat(encoder_hidden_states_enc_dec_loop, dim=0)

        # No need to modify scheduler_output_word
        # scheduler_output_byte needs to be modified

    def update_encoder_embeds_curr_word(self, exec_model_req: ExecuteModelRequest, 
                                        encoder_hidden_states: torch.Tensor, 
                                        word_lens_bytes_per_task: Optional[List[List[int]]] = None) -> None:
        """For all sequences in the batch, updates HATSequenceState.encoder_embeds_curr_word with the 
           encoder outputs for the current word.
        """
        assert not (self.is_prefill and word_lens_bytes_per_task is None)

        if self.is_prefill:
            offset = 0
            for idx, word_lens_bytes in enumerate(word_lens_bytes_per_task):
                offset += sum(word_lens_bytes[:-1])
                seq_id = next(iter(exec_model_req.seq_group_metadata_list[idx].seq_data.keys()))
                self.seq_ids_to_hat_state[seq_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states[offset:offset+word_lens_bytes[-1], :].clone()
                )
                offset += word_lens_bytes[-1]
        else:
            encoder_hidden_states_clone = encoder_hidden_states.clone()
            for idx in range(encoder_hidden_states.shape[0]):
                seq_id = next(iter(exec_model_req.seq_group_metadata_list[idx].seq_data.keys()))
                self.seq_ids_to_hat_state[seq_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states_clone[idx, :].unsqueeze(0)
                )
                
    def obtain_encoder_hidden_states_excl_last_word(self, encoder_hidden_states: torch.Tensor, 
                                                    word_lens_bytes_per_task: List[List[int]]) -> torch.Tensor:
        encoder_hidden_states_excl_last_word = (
            torch.zeros(sum(sum(word_lens_bytes[:-1]) for word_lens_bytes in word_lens_bytes_per_task), 
                        encoder_hidden_states.shape[1], 
                        dtype=encoder_hidden_states.dtype, device=self.device)
        )
        
        offset = 0
        offset_excl_last_word = 0
        for word_lens_bytes in word_lens_bytes_per_task:
            num_bytes_excl_last_word = sum(word_lens_bytes[:-1])
            num_bytes = num_bytes_excl_last_word + word_lens_bytes[-1]
            
            encoder_hidden_states_excl_last_word[offset_excl_last_word:offset_excl_last_word+num_bytes_excl_last_word, :] = encoder_hidden_states[offset:offset+num_bytes_excl_last_word, :]
            
            offset_excl_last_word += num_bytes_excl_last_word
            offset += num_bytes
            
        return encoder_hidden_states_excl_last_word

    def _add_new_sequence(self, new_req_data: NewRequestData, num_scheduled_tokens: int):
        """Initialises HATSequenceState for a new sequence and
           stores it in self.req_ids_to_hat_state, indexed by seq_id.

        Returns:
            List[int]: wordlens_bytes for the new sequence.
        """
        text_words_bytes = self._split_text(new_req_data.prompt_token_ids)
        
        curr_word_bytes = text_words_bytes[-1]
        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]

        self.req_ids_to_hat_state[new_req_data.req_id] = HATSequenceState(
            curr_word_bytes=curr_word_bytes,
            new_word_first_bytes=None,
            is_partial_prefill=len(new_req_data.prompt_token_ids) > num_scheduled_tokens,
            num_prompt_tokens=len(new_req_data.prompt_token_ids),
            num_computed_tokens_backbone=0,
            block_table_backbone=[],
            word_lens_bytes=word_lens_bytes,
            prev_pred_backbone_embedding=None,
            encoder_embeds_curr_word=[],
            encoder_embeds_new_word=[],
            word_position=torch.tensor(0, dtype=torch.int64, device=self.device),
            byte_position=0,
        )
        return word_lens_bytes
    
    def _update_sequence(self, cached_req_data: CachedRequestData):
        """Updates HATSequenceState for a cached sequence and
           stores it in self.req_ids_to_hat_state, indexed by seq_id.

        Returns:
            List[int]: wordlens_bytes for the new sequence.
        """
        seq_state = self.req_ids_to_hat_state[cached_req_data.req_id]

        text_words_bytes = self._split_text(seq_state.curr_word_bytes + cached_req_data.new_token_ids)
        curr_word_bytes = text_words_bytes[-1]
        seq_state.curr_word_bytes = curr_word_bytes

        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]
        seq_state.word_lens_bytes = word_lens_bytes

    def _split_text(self, text_bytes: List[int]) -> List[List[int]]:
        """Splits a text into its constituent words in bytes."""
        text = self.hat_splitter.decode(text_bytes, skip_special_tokens=False)
        list_of_words_in_bytes = self.hat_splitter.encode(text)
        return list_of_words_in_bytes