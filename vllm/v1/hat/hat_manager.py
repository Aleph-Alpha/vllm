import itertools
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sequence import ExecuteModelRequest
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.hat.hat_model_runner import HATModelRunner
from vllm.v1.hat.hat_splitter import HATRuleSplitter
from vllm.v1.hat.hat_utils import HATSequenceState, _create_empty_model_runner_output, _create_empty_scheduler_output, check_byte_for_new_word, safe_list_slice, safe_tensor_slice, split_text
from vllm.v1.outputs import ModelRunnerOutput


class HATManager:

    def __init__(self, 
                 special_token_dict: Dict[str, int],
                 max_word_size: int,
                 backbone_model_runner: HATModelRunner,
                 device: torch.device,
                 rank: int,
                 driver_rank: int):
        self.req_ids_to_hat_state: Dict[str, HATSequenceState] = {}
        self.hat_splitter = HATRuleSplitter(special_token_dict, max_word_size=max_word_size)
        self.backbone_model_runner = backbone_model_runner
        self.device = device
        self.rank = rank
        self.driver_rank = driver_rank
        
        self.num_decodes_not_word_boundary = 0
        self.num_decodes_word_boundary = 0
        
        # This will include prefills, last chunked prefill and decodes
        self.output: ModelRunnerOutput = _create_empty_model_runner_output()

        self.first_word_embedding: torch.Tensor = None
        
    def finish_step(self):
        self.num_decodes_not_word_boundary = 0
        self.num_decodes_word_boundary = 0
        tmp_output = self.output
        self.output = _create_empty_model_runner_output()
        return tmp_output if self.driver_rank == self.rank else None
    
    def add_request(self, scheduler_output: SchedulerOutput) -> Tuple[SchedulerOutput, SchedulerOutput]:

        scheduler_output_byte = _create_empty_scheduler_output()
        scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids
        scheduler_output_word = _create_empty_scheduler_output()
        scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids

        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            assert req_id not in self.req_ids_to_hat_state, f"Request {req_id} already exists in HATManager"

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_backbone = self._add_new_sequence(new_req_data, num_scheduled_tokens)
            req_state = self.req_ids_to_hat_state[req_id]

            # We need to split the block table into two for the encoder/decoder and the backbone
            block_table_enc_dec = new_req_data.block_ids[-1:]
            block_table_backbone = new_req_data.block_ids[:-1]
            # print("\n\nblock_table_enc_dec", block_table_enc_dec)
            # print("block_table_backbone", block_table_backbone)
            new_req_data.block_ids = block_table_enc_dec

            scheduler_output_byte.scheduled_new_reqs.append(new_req_data)
            scheduler_output_byte.num_scheduled_tokens[req_id] = num_scheduled_tokens
            scheduler_output_byte.total_num_scheduled_tokens += num_scheduled_tokens
            req_state.num_scheduled_tokens_byte = num_scheduled_tokens

            new_req_data_backbone = NewRequestData(
                req_id=req_id,
                prompt_token_ids=[0] * num_scheduled_tokens_backbone,
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=new_req_data.sampling_params,
                block_ids=block_table_backbone,
                num_computed_tokens=0,
                lora_request=new_req_data.lora_request,
            )

            if num_scheduled_tokens_backbone > 0:
                scheduler_output_word.scheduled_new_reqs.append(new_req_data_backbone)
                scheduler_output_word.num_scheduled_tokens[req_id] = num_scheduled_tokens_backbone
                scheduler_output_word.total_num_scheduled_tokens += num_scheduled_tokens_backbone
            else:
                # Need to manually register new requests that only contain an incomplete words with the backbone model runner
                # Because after the first step we will always get a CachedRequestState so the backbone model runner would not know
                # about it yet
                self.backbone_model_runner.register_request(new_req_data_backbone)
        
        decode_cached_reqs_not_word_boundary = []
        decode_cached_reqs_word_boundary = []
        decode_cached_reqs_word_boundary_backbone = []
        for cached_req_data in scheduler_output.scheduled_cached_reqs:
            req_id = cached_req_data.req_id
            assert req_id in self.req_ids_to_hat_state, f"Request {req_id} not found in HATManager"
            req_state = self.req_ids_to_hat_state[req_id]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # print("num_scheduled_tokens", num_scheduled_tokens)

            block_table_enc_dec = cached_req_data.new_block_ids[-1:]
            block_table_backbone = cached_req_data.new_block_ids[:-1]
            cached_req_data.new_block_ids = block_table_enc_dec
            
            # print("\n\nblock_table_enc_dec", block_table_enc_dec)
            # print("block_table_backbone", block_table_backbone)

            scheduler_output_byte.num_scheduled_tokens[req_id] = num_scheduled_tokens
            scheduler_output_byte.total_num_scheduled_tokens += num_scheduled_tokens
            scheduler_output_byte.finished_req_ids = scheduler_output.finished_req_ids
            req_state.num_scheduled_tokens_byte = num_scheduled_tokens

            # L TODO: This will be set once for NewRequestData to True (if it is partial prefill) and then at some point
            # changed to False in process_outputs when we processed the last partial prefill of that request
            if req_state.is_partial_prefill or cached_req_data.resumed_from_preemption:
                scheduler_output_byte.scheduled_cached_reqs.append(cached_req_data)

                if cached_req_data.resumed_from_preemption:
                    self._resume_preempted_sequence(cached_req_data, num_scheduled_tokens)
                else:
                    self._update_sequence(cached_req_data)

                word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes

                num_scheduled_tokens_backbone = len(word_lens_bytes) - 1
                req_state.num_scheduled_tokens_backbone = num_scheduled_tokens_backbone
                cached_req_data_backbone = CachedRequestData(
                    req_id=req_id,
                    resumed_from_preemption=cached_req_data.resumed_from_preemption,
                    new_token_ids=[0] * num_scheduled_tokens_backbone,
                    new_block_ids=block_table_backbone,
                    num_computed_tokens=self.req_ids_to_hat_state[req_id].num_computed_tokens_backbone,
                )

                if num_scheduled_tokens_backbone > 0:
                    scheduler_output_word.scheduled_cached_reqs.append(cached_req_data_backbone)
                    scheduler_output_word.num_scheduled_tokens[req_id] = num_scheduled_tokens_backbone
                    scheduler_output_word.total_num_scheduled_tokens += num_scheduled_tokens_backbone

                scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids
            # Decoder at word boundary
            elif len(req_state.new_word_first_bytes) > 0:
                self.req_ids_to_hat_state[req_id].word_lens_bytes = [1]

                # Update scheduler_output_word
                decode_cached_reqs_word_boundary.append(cached_req_data)
                cached_req_data_backbone = CachedRequestData(
                    req_id=req_id,
                    resumed_from_preemption=cached_req_data.resumed_from_preemption,
                    new_token_ids=[0],
                    new_block_ids=block_table_backbone,
                    num_computed_tokens=self.req_ids_to_hat_state[req_id].num_computed_tokens_backbone,
                )
                scheduler_output_word.num_scheduled_tokens[req_id] = 1
                scheduler_output_word.total_num_scheduled_tokens += 1
                scheduler_output_word.finished_req_ids = scheduler_output.finished_req_ids
                
                decode_cached_reqs_word_boundary_backbone.append(cached_req_data_backbone)
        
            else:
                self.req_ids_to_hat_state[req_id].word_lens_bytes = [1]
                decode_cached_reqs_not_word_boundary.append(cached_req_data)

        scheduler_output_byte.scheduled_cached_reqs.extend(decode_cached_reqs_word_boundary)
        scheduler_output_byte.scheduled_cached_reqs.extend(decode_cached_reqs_not_word_boundary)
        
        # Add in decodes that need the backbone
        scheduler_output_word.scheduled_cached_reqs.extend(decode_cached_reqs_word_boundary_backbone)
        
        self.num_decodes_not_word_boundary = len(decode_cached_reqs_not_word_boundary)
        self.num_decodes_word_boundary = len(decode_cached_reqs_word_boundary)
        
        return scheduler_output_byte, scheduler_output_word
    
    def remove_finished_requests(self, scheduler_output: SchedulerOutput):
        for req_id in scheduler_output.finished_req_ids:
            del self.req_ids_to_hat_state[req_id]

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
        
        encoder_hidden_states_final_decoder = safe_tensor_slice(encoder_hidden_states,
                                                                      self.num_decodes_not_word_boundary).clone()
        
        num_decodes = self.num_decodes_not_word_boundary + self.num_decodes_word_boundary
        encoder_hidden_states_decodes = safe_tensor_slice(encoder_hidden_states,
                                                                num_decodes,
                                                                keep_prefix=False).clone()

        offset = 0
        offset_beginning = 0
        
        # Mutually exclusive.
        scheduler_output_byte_enc_dec = _create_empty_scheduler_output()
        scheduler_output_byte_final_decoder = _create_empty_scheduler_output()
        decodes = 0
        for scheduled_req in scheduler_output_byte.scheduled_new_reqs + scheduler_output_byte.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            word_lens_bytes = self.req_ids_to_hat_state[req_id].word_lens_bytes
            req_state = self.req_ids_to_hat_state[req_id]
            # Case where we have a sequence shorter than a full word
            if req_state.is_partial_prefill:
                num_bytes_excl_last_word = 0
                if req_state.num_scheduled_tokens_backbone > 0:
                    # We remove the overcounting in word_len_bytes due to the last computed work in the previous chunked prefill 
                    num_bytes_excl_last_word = sum(word_lens_bytes[:-1])
                    encoder_hidden_states_encoder_connector.extend(self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word)
                    encoder_hidden_states_encoder_connector.append(
                        encoder_hidden_states[offset_beginning:offset_beginning+num_bytes_excl_last_word, :]
                    )
                    offset += num_bytes_excl_last_word

                    # Usually encoder_embeds_new_word will be empty for chunked prefills except for the special case
                    # where the chunked prefill occurs inside a multi-byte character split
                    self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word = self.req_ids_to_hat_state[req_id].encoder_embeds_new_word
                    self.req_ids_to_hat_state[req_id].encoder_embeds_new_word = []

                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states[offset:offset+word_lens_bytes[-1], :].clone()
                )
                    
                offset += word_lens_bytes[-1]
                offset_beginning += num_bytes_excl_last_word + word_lens_bytes[-1]
                
                if isinstance(scheduled_req, NewRequestData):
                    scheduler_output_byte_final_decoder.scheduled_new_reqs.append(scheduled_req)
                else:
                    scheduler_output_byte_final_decoder.scheduled_cached_reqs.append(scheduled_req)

                scheduler_output_byte_final_decoder.num_scheduled_tokens[req_id] = req_state.num_scheduled_tokens_byte
                scheduler_output_byte_final_decoder.total_num_scheduled_tokens += req_state.num_scheduled_tokens_byte
                scheduler_output_byte_final_decoder.finished_req_ids = scheduler_output_byte.finished_req_ids
            elif req_state.is_prefill:
                num_bytes_excl_last_word = sum(word_lens_bytes[:-1])
                offset += num_bytes_excl_last_word
                self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                    encoder_hidden_states[offset:offset+word_lens_bytes[-1], :].clone()
                )

                if req_state.num_scheduled_tokens_backbone > 0:
                    encoder_hidden_states_encoder_connector.append(
                        encoder_hidden_states[offset_beginning:offset_beginning+num_bytes_excl_last_word, :]
                    )
                else:
                    assert num_bytes_excl_last_word == 0

                offset += word_lens_bytes[-1]
                offset_beginning += num_bytes_excl_last_word + word_lens_bytes[-1]
                
                if isinstance(scheduled_req, NewRequestData):
                    scheduler_output_byte_final_decoder.scheduled_new_reqs.append(scheduled_req)
                else:
                    scheduler_output_byte_final_decoder.scheduled_cached_reqs.append(scheduled_req)
                scheduler_output_byte_final_decoder.num_scheduled_tokens[req_id] = req_state.num_scheduled_tokens_byte
                scheduler_output_byte_final_decoder.total_num_scheduled_tokens += req_state.num_scheduled_tokens_byte
                scheduler_output_byte_final_decoder.finished_req_ids = scheduler_output_byte.finished_req_ids
            else:
                assert isinstance(scheduled_req, CachedRequestData)

                # Decodes that at the start of the loop already reached word boundary do not
                # go into enc/dec loop
                if len(req_state.new_word_first_bytes) > 0:
                    self.req_ids_to_hat_state[req_id].encoder_embeds_new_word.append(
                        encoder_hidden_states_decodes[decodes, :].unsqueeze(0)
                    )
                    
                    scheduler_output_byte_final_decoder.scheduled_cached_reqs.append(scheduled_req)
                    scheduler_output_byte_final_decoder.num_scheduled_tokens[req_id] = 1
                    scheduler_output_byte_final_decoder.total_num_scheduled_tokens += 1
                    scheduler_output_byte_final_decoder.finished_req_ids = scheduler_output_byte.finished_req_ids
                else:
                    # encoder_embeds_curr_word and curr_word_bytes are now synced
                    self.req_ids_to_hat_state[req_id].encoder_embeds_curr_word.append(
                        encoder_hidden_states_decodes[decodes, :].unsqueeze(0)
                    )
                    encoder_hidden_states_enc_dec_loop.append(encoder_hidden_states[offset, :].unsqueeze(0))

                    scheduler_output_byte_enc_dec.scheduled_cached_reqs.append(scheduled_req)
                    scheduler_output_byte_enc_dec.num_scheduled_tokens[req_id] = 1
                    scheduler_output_byte_enc_dec.total_num_scheduled_tokens += 1
                    scheduler_output_byte_enc_dec.finished_req_ids = scheduler_output_byte.finished_req_ids
                    
                decodes += 1
                offset += 1
                offset_beginning += 1                
        
        if len(encoder_hidden_states_enc_dec_loop) > 0:
            encoder_hidden_states_enc_dec_loop = torch.cat(encoder_hidden_states_enc_dec_loop, dim=0).clone()
        else:
            encoder_hidden_states_enc_dec_loop = None

        return (
            encoder_hidden_states_encoder_connector, 
            encoder_hidden_states_enc_dec_loop, 
            encoder_hidden_states_final_decoder, 
            scheduler_output_byte_enc_dec,
            scheduler_output_byte_final_decoder
        )
        
    def handle_encoder_output_loop(self, encoder_hidden_states_enc_dec_loop: torch.Tensor,
                                   scheduler_output_byte_enc_dec: SchedulerOutput):
        encoder_hidden_states = encoder_hidden_states_enc_dec_loop.clone()
        for idx, scheduled_req in enumerate(scheduler_output_byte_enc_dec.scheduled_cached_reqs):
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            req_state.encoder_embeds_curr_word.append(
                encoder_hidden_states[idx, :].unsqueeze(0)
            )
    
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
                # print("req_id", req_id, "req_state.word_lens_bytes", req_state.word_lens_bytes)
                
                # For encoder connector, we need to include the last possibly incomplete word 
                # the previous worker step for chunked prefills
                # P WE SHOULD NOT DO THIS FOR FIRST CHUNKED PREFILL
                word_lens_bytes_per_task_excl_last_word[-1][0] += req_state.len_last_word_chunked

                num_bytes_last_word = req_state.word_lens_bytes[-1]
                byte_positions.append(torch.arange(req_state.byte_position - req_state.len_last_word_chunked - req_state.multi_bytes,
                                                   req_state.byte_position + req_state.num_scheduled_tokens_byte - num_bytes_last_word - req_state.multi_bytes,
                                                   device=self.device))
                word_positions.append(torch.arange(req_state.word_position_cpu,
                                                   req_state.word_position_cpu + scheduler_output_word.num_scheduled_tokens[req_id],
                                                   device=self.device))
            elif req_state.is_prefill:
                # word_lens_bytes_per_task.append(req_state.word_lens_bytes)
                num_bytes_last_word = req_state.word_lens_bytes[-1]
                word_lens_bytes_per_task_excl_last_word.append(req_state.word_lens_bytes[:-1])
                byte_positions.append(torch.arange(0, req_state.num_scheduled_tokens_byte - num_bytes_last_word, device=self.device))
                word_positions.append(torch.arange(0, scheduler_output_word.num_scheduled_tokens[req_id], device=self.device))
            else:
                # word_len_bytes_per_task only needed for final decoder
                word_lens_bytes_per_task_excl_last_word.append([len(req_state.curr_word_bytes)])
                byte_positions.append(torch.arange(req_state.byte_position - len(req_state.curr_word_bytes),
                                                   req_state.byte_position,
                                                   device=self.device))
                word_positions.append(req_state.word_position)
                
                encoder_hidden_states_encoder_connector.extend(req_state.encoder_embeds_curr_word)                

        # Assume for now that all decodes finish at a word boundary
        encoder_hidden_states_encoder_connector = torch.cat(encoder_hidden_states_encoder_connector, dim=0)
        word_lens_bytes_per_task_excl_last_word = torch.tensor(list(itertools.chain.from_iterable(word_lens_bytes_per_task_excl_last_word)),
                                                               dtype=torch.int32,
                                                               device=self.device)
        byte_positions = torch.hstack(byte_positions)
        word_positions = torch.hstack(word_positions)

        return encoder_hidden_states_encoder_connector, word_lens_bytes_per_task_excl_last_word, byte_positions, word_positions
    
    def handle_backbone_output(self, scheduler_output_byte_final_decoder: SchedulerOutput,
                               predictive_word_embeddings: torch.Tensor) -> torch.Tensor:
        # L TODO: self.num_decodes might not be correct here when we later run for a static num of bytes
        predictive_word_embeddings_decodes = None
        if predictive_word_embeddings is not None:
            predictive_word_embeddings_decodes = safe_tensor_slice(predictive_word_embeddings,
                                                                self.num_decodes_word_boundary,
                                                                keep_prefix=False).clone()
        predictive_word_embeddings_final_decoder = []

        num_decodes = 0
        offset = 0
        for scheduled_req in scheduler_output_byte_final_decoder.scheduled_new_reqs + scheduler_output_byte_final_decoder.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            if req_state.is_partial_prefill:
                num_words_excl_last_word = len(req_state.word_lens_bytes) - 1
                
                # For first chunked prefill we append first word embedding
                # If not, we append the previous predicted word embedding from previous worker step
                predictive_word_embeddings_final_decoder.append(req_state.prev_pred_backbone_embedding)
                    
                if req_state.num_scheduled_tokens_backbone > 0:
                    predictive_word_embeddings_final_decoder.append(predictive_word_embeddings[offset:offset+num_words_excl_last_word, :])
                    
                    offset += num_words_excl_last_word
                    req_state.prev_pred_backbone_embedding = predictive_word_embeddings[offset-1, :].clone().unsqueeze(0)
                    
            elif req_state.is_prefill:
                num_words_excl_last_word = len(req_state.word_lens_bytes) - 1
                predictive_word_embeddings_final_decoder.append(self.first_word_embedding)
                
                if req_state.num_scheduled_tokens_backbone > 0:
                    predictive_word_embeddings_final_decoder.append(predictive_word_embeddings[offset:offset+num_words_excl_last_word, :])
                
                    offset += num_words_excl_last_word
                    req_state.prev_pred_backbone_embedding = predictive_word_embeddings[offset-1, :].clone().unsqueeze(0)
                 
            else:
                req_state.prev_pred_backbone_embedding = predictive_word_embeddings_decodes[num_decodes, :].unsqueeze(0)
                predictive_word_embeddings_final_decoder.append(predictive_word_embeddings[offset, :].unsqueeze(0))
                num_decodes += 1
                offset += 1
                
                
        predictive_word_embeddings_final_decoder = torch.cat(predictive_word_embeddings_final_decoder, dim=0)
        return predictive_word_embeddings_final_decoder
    
    def prepare_input_final_decoder(self, scheduler_output_byte_final_decoder: SchedulerOutput) -> torch.Tensor: 
        word_lens_bytes_per_task = []
        for scheduled_req in scheduler_output_byte_final_decoder.scheduled_new_reqs + scheduler_output_byte_final_decoder.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            word_lens_bytes_per_task.append(req_state.word_lens_bytes)
            
        word_lens_bytes_per_task = torch.tensor(list(itertools.chain.from_iterable(word_lens_bytes_per_task)),
                                                dtype=torch.int32).pin_memory()
        word_lens_bytes_per_task = word_lens_bytes_per_task.to(self.device, non_blocking=True)
        
        return word_lens_bytes_per_task
    
    def prepare_exec_model_req_for_dec_autoregressive_phase(self, scheduler_output_byte_enc_dec: SchedulerOutput) -> torch.Tensor:
        predictive_word_embeddings = [] 
        for scheduled_req in scheduler_output_byte_enc_dec.scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            predictive_word_embeddings.append(req_state.prev_pred_backbone_embedding)
        
        predictive_word_embeddings = torch.cat(predictive_word_embeddings, dim=0)
        return predictive_word_embeddings

    def update_backbone_info(self, scheduler_output_word: SchedulerOutput):
        cached_reqs = safe_list_slice(scheduler_output_word.scheduled_cached_reqs, 
                                      self.num_decodes_word_boundary,
                                      keep_prefix=False)
        for scheduled_req in cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            req_state.num_computed_tokens_backbone += 1
            req_state.word_position += 1
            req_state.word_position_cpu += 1
            
            req_state.encoder_embeds_curr_word = req_state.encoder_embeds_new_word
            req_state.encoder_embeds_new_word = []
            
            req_state.curr_word_bytes = req_state.new_word_first_bytes
            req_state.new_word_first_bytes = []
    
    def process_outputs_prefill_chunked_prefill(self, scheduler_output_byte_final_decoder: SchedulerOutput, model_runner_output: ModelRunnerOutput):
        cached_reqs = safe_list_slice(scheduler_output_byte_final_decoder.scheduled_cached_reqs, 
                                      self.num_decodes_word_boundary)
        for scheduled_req in scheduler_output_byte_final_decoder.scheduled_new_reqs + cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]
            assert req_id not in self.output.req_id_to_index

            self.output.req_id_to_index[req_id] = len(self.output.req_ids) 
            self.output.req_ids.append(req_id)
            self.output.sampled_token_ids.append([])
            self.output.prompt_logprobs_dict[req_id] = None

            new_token_id = None
            # If we are in prefill case or last chunked prefill 
            if not req_state.is_partial_prefill or req_state.num_prompt_tokens == (req_state.num_scheduled_tokens_byte + scheduled_req.num_computed_tokens): 
                req_id_index_step = model_runner_output.req_id_to_index[req_id]
                sampled_token_ids: List[int] = model_runner_output.sampled_token_ids[req_id_index_step]
                new_token_id = sampled_token_ids[0]
                self.output.sampled_token_ids[-1].append(new_token_id)
                req_state.is_partial_prefill = False
                req_state.is_prefill = False

            req_state.byte_position += req_state.num_scheduled_tokens_byte
            req_state.word_position += req_state.num_scheduled_tokens_backbone
            req_state.word_position_cpu += req_state.num_scheduled_tokens_backbone
            req_state.num_computed_tokens_backbone += req_state.num_scheduled_tokens_backbone

            if req_state.is_partial_prefill:
                continue
                
            # Prefill case or last chunked prefill            
            curr_word_bytes = req_state.curr_word_bytes
            curr_word_bytes.append(new_token_id)

            if new_token_id == self.hat_splitter.eot_id:                
                continue

            is_new_word, words = check_byte_for_new_word(self.hat_splitter, curr_word_bytes)
            if is_new_word:
                req_state.curr_word_bytes = words[0]
                req_state.new_word_first_bytes = words[1]
                
                len_new_word = len(words[1])
                # Multi byte characters
                if len_new_word > 1:
                    req_state.encoder_embeds_new_word = req_state.encoder_embeds_curr_word[-len_new_word+1:]
                    req_state.encoder_embeds_curr_word = req_state.encoder_embeds_curr_word[:-len_new_word+1]
                
    def process_outputs_enc_dec_loop(self, scheduled_cached_reqs: List[CachedRequestData], model_runner_output: ModelRunnerOutput) -> SchedulerOutput:
        """
        - Take out decodes which have reached word boundary from SchedulerOutput. This means updating `scheduled_cached_reqs`, `num_scheduled_tokens`
        `total_num_scheduled_tokens`
        """
        scheduler_output_byte_enc_dec_running_tmp = _create_empty_scheduler_output()
        # scheduler_output_byte_enc_dec_running_tmp.finished_req_ids = scheduler_output_byte_enc_dec_running.finished_req_ids

        for scheduled_req in scheduled_cached_reqs:
            req_id = scheduled_req.req_id
            req_state = self.req_ids_to_hat_state[req_id]

            if req_id not in self.output.req_id_to_index:
                self.output.req_id_to_index[req_id] = len(self.output.req_ids) 
                self.output.req_ids.append(req_id)
                req_id_index_step = model_runner_output.req_id_to_index[req_id]
                
                sampled_token_ids: List[int] = model_runner_output.sampled_token_ids[req_id_index_step]
                new_token_id = sampled_token_ids[0]
                self.output.sampled_token_ids.append([new_token_id])
                self.output.prompt_logprobs_dict[req_id] = None
            else:
                # self.output.req_id_to_index contains info for all seqs in this worker step
                # model_runner_output only contains info about seq currently running in the loop
                req_id_index_output = self.output.req_id_to_index[req_id]
                req_id_index_step = model_runner_output.req_id_to_index[req_id]

                sampled_token_ids: List[int] = model_runner_output.sampled_token_ids[req_id_index_step]
                new_token_id = sampled_token_ids[-1]
                self.output.sampled_token_ids[req_id_index_output].append(new_token_id)

            req_state.byte_position += 1
            curr_word_bytes = req_state.curr_word_bytes
            curr_word_bytes.append(new_token_id)

            if new_token_id == self.hat_splitter.eot_id:                
                continue

            is_new_word, words = check_byte_for_new_word(self.hat_splitter, curr_word_bytes)
            if is_new_word:
                req_state.curr_word_bytes = words[0]
                req_state.new_word_first_bytes = words[1]
                
                len_new_word = len(words[1])
                # Multi byte characters
                if len_new_word > 1:
                    req_state.encoder_embeds_new_word = req_state.encoder_embeds_curr_word[-len_new_word+1:]
                    req_state.encoder_embeds_curr_word = req_state.encoder_embeds_curr_word[:-len_new_word+1]
                    
            else:
                cached_req = CachedRequestData(
                    req_id=req_id,
                    resumed_from_preemption=False,
                    new_token_ids=[new_token_id],
                    new_block_ids=[[]],
                    num_computed_tokens=scheduled_req.num_computed_tokens + 1,
                )
                scheduler_output_byte_enc_dec_running_tmp.scheduled_cached_reqs.append(cached_req)
                scheduler_output_byte_enc_dec_running_tmp.num_scheduled_tokens[req_id] = 1
                scheduler_output_byte_enc_dec_running_tmp.total_num_scheduled_tokens += 1

        self.num_decodes_not_word_boundary = len(scheduler_output_byte_enc_dec_running_tmp.scheduled_cached_reqs)
        return scheduler_output_byte_enc_dec_running_tmp
        
    def _add_new_sequence(self, new_req_data: NewRequestData, num_scheduled_tokens: int):
        """Initialises HATSequenceState for a new sequence and
           stores it in self.req_ids_to_hat_state, indexed by seq_id.

        Returns:
            List[int]: wordlens_bytes for the new sequence.
        """
        text_words_bytes = split_text(self.hat_splitter, new_req_data.prompt_token_ids)
        
        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]

        is_partial_prefill = len(new_req_data.prompt_token_ids) > num_scheduled_tokens
        
        cu_word_lens_bytes = torch.cumsum(torch.tensor(word_lens_bytes), dim=0)
        num_scheduled_tokens_backbone = int(torch.searchsorted(cu_word_lens_bytes,
                                                                num_scheduled_tokens,
                                                                right=False).item())
        word_lens_bytes = word_lens_bytes[:num_scheduled_tokens_backbone+1]
        
        # curr_word_byes should be obtained from word_lens_bytes
        offset = cu_word_lens_bytes[num_scheduled_tokens_backbone] - num_scheduled_tokens
        offset = int(offset.item())
        word_lens_bytes[-1] -= offset
        curr_word_bytes = safe_list_slice(text_words_bytes[num_scheduled_tokens_backbone], offset)

        self.req_ids_to_hat_state[new_req_data.req_id] = HATSequenceState(
            curr_word_bytes=curr_word_bytes,
            new_word_first_bytes=[],
            is_partial_prefill=is_partial_prefill,
            is_prefill=True,
            num_prompt_tokens=len(new_req_data.prompt_token_ids),
            num_computed_tokens_backbone=0,
            num_scheduled_tokens_byte=num_scheduled_tokens,
            num_scheduled_tokens_backbone=num_scheduled_tokens_backbone,
            len_last_word_chunked=0,
            multi_bytes=0,
            word_lens_bytes=word_lens_bytes,
            prev_pred_backbone_embedding=self.first_word_embedding,
            encoder_embeds_curr_word=[],
            encoder_embeds_new_word=[],
            word_position=torch.tensor(0, dtype=torch.int64, device=self.device),
            word_position_cpu=0,
            byte_position=0,
        )
        return num_scheduled_tokens_backbone
    
    def _resume_preempted_sequence(self, cached_req_data: CachedRequestData, num_scheduled_tokens: int):
        """Initialises HATSequenceState for a preempted sequence

        Returns:
            List[int]: wordlens_bytes for the preempted sequence.
        """
        req_id = cached_req_data.req_id
        req_state = self.req_ids_to_hat_state[req_id]
        
        text_words_bytes = split_text(self.hat_splitter, cached_req_data.new_token_ids)
        
        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]

        num_prompt_tokens = max(req_state.num_prompt_tokens, req_state.byte_position)
        is_partial_prefill = num_prompt_tokens > num_scheduled_tokens
        
        cu_word_lens_bytes = torch.cumsum(torch.tensor(word_lens_bytes), dim=0)
        num_scheduled_tokens_backbone = int(torch.searchsorted(cu_word_lens_bytes,
                                                                num_scheduled_tokens,
                                                                right=False).item())
        word_lens_bytes = word_lens_bytes[:num_scheduled_tokens_backbone+1]
        
        # curr_word_byes should be obtained from word_lens_bytes
        offset = cu_word_lens_bytes[num_scheduled_tokens_backbone] - num_scheduled_tokens
        offset = int(offset.item())
        word_lens_bytes[-1] -= offset
        curr_word_bytes = safe_list_slice(text_words_bytes[num_scheduled_tokens_backbone], offset)

        self.req_ids_to_hat_state[req_id] = HATSequenceState(
            curr_word_bytes=curr_word_bytes,
            new_word_first_bytes=[],
            is_partial_prefill=is_partial_prefill,
            is_prefill=not is_partial_prefill,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens_backbone=0,
            num_scheduled_tokens_byte=num_scheduled_tokens,
            num_scheduled_tokens_backbone=num_scheduled_tokens_backbone,
            len_last_word_chunked=0,
            multi_bytes=0,
            word_lens_bytes=word_lens_bytes,
            prev_pred_backbone_embedding=self.first_word_embedding,
            encoder_embeds_curr_word=[],
            encoder_embeds_new_word=[],
            word_position=torch.tensor(0, dtype=torch.int64, device=self.device),
            word_position_cpu=0,
            byte_position=0,
        )
        return num_scheduled_tokens_backbone
    
    def _update_sequence(self, cached_req_data: CachedRequestData):
        """Updates HATSequenceState for a cached sequence and
           stores it in self.req_ids_to_hat_state, indexed by seq_id.

        Returns:
            List[int]: wordlens_bytes for the new sequence.
        """
        req_state = self.req_ids_to_hat_state[cached_req_data.req_id]

        text_words_bytes = split_text(self.hat_splitter, req_state.curr_word_bytes + cached_req_data.new_token_ids)
        
        # word_lens_bytes always only includes characters from this worker step
        word_lens_bytes = [len(text_word_bytes) for text_word_bytes in text_words_bytes]

        # Rare edge case where the chunked prefill occurs inside a multi-byte character split
        # e.g. "🤔🤔🤔" and the chunked prefill has size 5
        if len(req_state.curr_word_bytes) > word_lens_bytes[0] and len(word_lens_bytes) == 2:
            multi_bytes = len(req_state.curr_word_bytes) - word_lens_bytes[0]
            len_last_word_chunked = word_lens_bytes[0]
            
            word_lens_bytes[0] = 0
            word_lens_bytes[1] = len(cached_req_data.new_token_ids)
            
            curr_word_bytes = text_words_bytes[-1]
            full_curr_word_embeds = torch.cat(req_state.encoder_embeds_curr_word, dim=0)
            req_state.encoder_embeds_new_word = [full_curr_word_embeds[-multi_bytes:]]
            req_state.encoder_embeds_curr_word = [full_curr_word_embeds[:-multi_bytes]]
        else:
            multi_bytes = 0
            len_last_word_chunked = len(req_state.curr_word_bytes)
            word_lens_bytes[0] -= len(req_state.curr_word_bytes)
            curr_word_bytes = text_words_bytes[-1]
            
        req_state.word_lens_bytes = word_lens_bytes
        
        # We overwrite curr_word_bytes with info from this worker step
        # encoder_curr_embeds still contains last possibly incomplete word 
        # from previous worker step
        req_state.len_last_word_chunked = len_last_word_chunked
        req_state.curr_word_bytes = curr_word_bytes
        req_state.multi_bytes = multi_bytes