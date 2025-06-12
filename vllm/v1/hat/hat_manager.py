import itertools
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sequence import ExecuteModelRequest
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.hat.hat_utils import HATSequenceState, _create_empty_scheduler_output
from vllm.v1.outputs import SamplerOutput


class HATManager:

    def __init__(self, special_token_dict: Dict[str, int], max_word_size: int, device: torch.device, rank: int, driver_rank: int):
        self.req_ids_to_hat_state: Dict[str, HATSequenceState] = {}
        self.hat_splitter = HATRuleSplitter(special_token_dict, max_word_size=max_word_size)
        self.device = device
        self.rank = rank
        self.driver_rank = driver_rank
        
        self.num_decodes = 0
        self.outputs: List[SamplerOutput] = []

        self.first_word_embedding: torch.Tensor = None
        
    def reset_manager(self):
        self.outputs = []
        self.num_decodes = 0
    
    def add_request(self, scheduler_output: SchedulerOutput) -> Tuple[SchedulerOutput, SchedulerOutput]:

        scheduler_output_byte = _create_empty_scheduler_output()
        scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids
        scheduler_output_word = _create_empty_scheduler_output()
        scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids

        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            assert req_id not in self.req_ids_to_hat_state, f"Request {req_id} already exists in HATManager"

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            self._add_new_sequence(new_req_data, num_scheduled_tokens)
            req_state = self.req_ids_to_hat_state[req_id]
            word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes

            # We need to split the block table into two for the encoder/decoder and the backbone
            block_table_enc_dec = new_req_data.block_ids[-1:]
            block_table_backbone = new_req_data.block_ids[:-1]
            new_req_data.block_ids = block_table_enc_dec

            scheduler_output_byte.scheduled_new_reqs.append(new_req_data)
            scheduler_output_byte.num_scheduled_tokens[req_id] = num_scheduled_tokens
            scheduler_output_byte.total_num_scheduled_tokens += num_scheduled_tokens
            req_state.num_scheduled_tokens_byte = num_scheduled_tokens

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
        
        #print(scheduler_output_byte)
        #print(scheduler_output_word)

        decode_cached_reqs_data = []
        for cached_req_data in scheduler_output.scheduled_cached_reqs:
            req_id = cached_req_data.req_id
            assert req_id in self.req_ids_to_hat_state, f"Request {req_id} not found in HATManager"
            req_state = self.req_ids_to_hat_state[req_id]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            block_table_enc_dec = cached_req_data.new_block_ids[-1:]
            block_table_backbone = cached_req_data.new_block_ids[:-1]
            req_state.block_table_backbone = block_table_backbone
            cached_req_data.new_block_ids = block_table_enc_dec

            scheduler_output_byte.num_scheduled_tokens[req_id] = num_scheduled_tokens
            scheduler_output_byte.total_num_scheduled_tokens += num_scheduled_tokens
            scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids
            req_state.num_scheduled_tokens_byte = num_scheduled_tokens

            # L TODO: This will be set once for NewRequestData to True (if it is partial prefill) and then at some point
            # changed to False in process_outputs when we processed the last partial prefill of that request
            if req_state.is_partial_prefill:
                scheduler_output_byte.scheduled_cached_reqs.append(cached_req_data)

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
                decode_cached_reqs_data.append(cached_req_data)

        scheduler_output_byte.scheduled_cached_reqs.extend(decode_cached_reqs_data)
        self.num_decodes = len(decode_cached_reqs_data)
        return scheduler_output_byte, scheduler_output_word
    
    def handle_encoder_output(self, scheduler_output_byte: SchedulerOutput,
                              encoder_hidden_states: torch.Tensor) -> Tuple[List[torch.Tensor],
                                                                            Optional[torch.Tensor],
                                                                            torch.Tensor,
                                                                            SchedulerOutput]:
        # Contains encoder hidden states for prefills and chunked_prefills excluding the last word
        # We will later add in the the encoder_embeds for decode sequences that reach the word boundary
        encoder_hidden_states_encoder_connector =[]
        # Contains encoder hidden state for all decode sequences
        encoder_hidden_states_enc_dec_loop = []

        # Contains encoder hidden states for prefill and chunked_prefills. 
        # Necessary for final decoder forward pass
        encoder_hidden_states_final_decoder = encoder_hidden_states[:-self.num_decodes, :]
        # Want to save in HATSequenceState.curr_word_embeds the embed we get from first encoder forward pass
        # This is to avoid cloning for every decode sequence
        encoder_hidden_states_decodes = encoder_hidden_states[-self.num_decodes:, :].clone()

        offset = 0
        offset_beginning = 0
        scheduler_output_byte_enc_dec = _create_empty_scheduler_output()
        decodes = 0
        for scheduled_req in scheduler_output_byte.scheduled_new_reqs + scheduler_output_byte.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes
            req_state = self.req_ids_to_hat_state[req_id]
            if req_state.is_partial_prefill:
                # We remove the overcounting in word_len_bytes due to the last computed work in the previous chunked prefill 
                num_bytes_excl_last_word = sum(word_lens_bytes[:-1])
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
                assert isinstance(scheduled_req, CachedRequestData)

                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states_decodes[decodes, :].unsqueeze(0)
                )
                encoder_hidden_states_enc_dec_loop.append(encoder_hidden_states[offset, :].unsqueeze(0))
                offset += 1
                offset_beginning += 1
                decodes += 1

                scheduler_output_byte_enc_dec.scheduled_cached_reqs.append(scheduled_req)
                scheduler_output_byte_enc_dec.num_scheduled_tokens[req_id] = 1
                scheduler_output_byte_enc_dec.total_num_scheduled_tokens += 1
                scheduler_output_byte_enc_dec.finished_req_ids = scheduler_output_byte.finished_req_ids
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
        
        if len(encoder_hidden_states_enc_dec_loop) > 0:
            encoder_hidden_states_enc_dec_loop = torch.cat(encoder_hidden_states_enc_dec_loop, dim=0)
        else:
            encoder_hidden_states_enc_dec_loop = None

        return (
            encoder_hidden_states_encoder_connector, 
            encoder_hidden_states_enc_dec_loop, 
            encoder_hidden_states_final_decoder, 
            scheduler_output_byte_enc_dec
        )

    def combine_scheduler_outputs(self, scheduler_output_word: SchedulerOutput,
                                  scheduler_output_byte_enc_dec_word_boundary: SchedulerOutput) -> SchedulerOutput:
        for cached_req_data in scheduler_output_byte_enc_dec_word_boundary.scheduled_cached_reqs:
            req_id = cached_req_data.req_id

            cached_req_data_backbone = CachedRequestData(
                req_id=req_id,
                resumed_from_preemption=cached_req_data.resumed_from_preemption,
                new_token_ids=[0],
                new_block_ids=self.req_ids_to_hat_state[req_id].block_table_backbone,
                num_computed_tokens=self.req_ids_to_hat_state[req_id].num_computed_tokens_backbone,
            )

            scheduler_output_word.scheduled_cached_reqs.append(cached_req_data_backbone)
            scheduler_output_word.num_scheduled_tokens[req_id] = 1
            scheduler_output_word.total_num_scheduled_tokens += 1
            scheduler_output_word.finished_req_ids = scheduler_output_word.finished_req_ids

        return scheduler_output_word
    
    def prepare_input_encoder_connector(self, encoder_hidden_states_encoder_connector: List[torch.Tensor],
                                        scheduler_output_word: SchedulerOutput):
        # word_lens_bytes_per_task = []
        # Only needed for encoder connector
        # For chunked prefill, we do need to include curr_word_bytes
        word_lens_bytes_per_task_excl_last_word = [[0]]
        byte_positions = []
        word_positions = []

        for scheduled_req in scheduler_output_word.scheduled_new_reqs + scheduler_output_word.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            if req_state.is_partial_prefill:
                # word_lens_bytes_per_task.append(req_state.word_lens_bytes)
                word_lens_bytes_per_task_excl_last_word.append(req_state.word_lens_bytes[:-1])
                # For encoder connector, we need to include the last possibly incomplete word 
                # the previous worker step for chunked prefills
                word_lens_bytes_per_task_excl_last_word[-1][0] += len(req_state.encoder_embeds_curr_word)
                num_bytes_last_word = req_state.word_lens_bytes[-1]
                byte_positions.append(torch.arange(req_state.byte_position,
                                                   req_state.byte_position + req_state.num_scheduled_tokens_byte - num_bytes_last_word,
                                                   device=self.device))
                word_positions.append(torch.arange(req_state.word_position_cpu,
                                                   req_state.word_position_cpu + scheduler_output_word.num_scheduled_tokens[req_id],
                                                   device=self.device))
            elif len(req_state.word_lens_bytes) == 1 and req_state.word_lens_bytes[0] == 1:
                # word_len_bytes_per_task only needed for final decoder
                word_lens_bytes_per_task_excl_last_word.append([len(req_state.encoder_embeds_curr_word)])
                byte_positions.append(torch.arange(req_state.byte_position - len(req_state.encoder_embeds_curr_word),
                                                   req_state.byte_position,
                                                   device=self.device))
                word_positions.append(req_state.word_position)
                encoder_hidden_states_encoder_connector.append(req_state.encoder_embeds_curr_word)
            else:
                # word_lens_bytes_per_task.append(req_state.word_lens_bytes)
                num_bytes_last_word = req_state.word_lens_bytes[-1]
                word_lens_bytes_per_task_excl_last_word.append(req_state.word_lens_bytes[:-1])
                byte_positions.append(torch.arange(0, req_state.num_scheduled_tokens_byte - num_bytes_last_word, device=self.device))
                word_positions.append(torch.arange(0, scheduler_output_word.num_scheduled_tokens[req_id], device=self.device))

        # Assume for now that all decodes finish at a word boundary
        encoder_hidden_states_encoder_connector = torch.cat(encoder_hidden_states_encoder_connector, dim=0)
        word_lens_bytes_per_task_excl_last_word = torch.tensor(list(itertools.chain.from_iterable(word_lens_bytes_per_task_excl_last_word)),
                                                               dtype=torch.int32,
                                                               device=self.device)
        byte_positions = torch.hstack(byte_positions)
        word_positions = torch.hstack(word_positions)

        return encoder_hidden_states_encoder_connector, word_lens_bytes_per_task_excl_last_word, byte_positions, word_positions

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
            num_scheduled_tokens_byte=num_scheduled_tokens,
            block_table_backbone=[],
            word_lens_bytes=word_lens_bytes,
            prev_pred_backbone_embedding=None,
            encoder_embeds_curr_word=[],
            encoder_embeds_new_word=[],
            word_position=torch.tensor(0, dtype=torch.int64, device=self.device),
            word_position_cpu=0,
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
        # We overwrite curr_word_bytes with info from this worker step
        # encoder_curr_embeds still contains last possibly incomplete word 
        # from previous worker step
        seq_state.curr_word_bytes = curr_word_bytes

        # word_lens_bytes always only includes characters from this worker step
        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]
        word_lens_bytes[0] -= len(seq_state.encoder_embeds_curr_word)
        seq_state.word_lens_bytes = word_lens_bytes

    def _split_text(self, text_bytes: List[int]) -> List[List[int]]:
        """Splits a text into its constituent words in bytes."""
        text = self.hat_splitter.decode(text_bytes, skip_special_tokens=False)
        list_of_words_in_bytes = self.hat_splitter.encode(text)
        return list_of_words_in_bytes