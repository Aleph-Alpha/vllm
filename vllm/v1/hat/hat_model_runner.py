# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import time
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (GraphCaptureContext, get_pp_group,
                                             get_tp_group, graph_capture, is_global_first_rank)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.hat import (HATBackboneForCausalLM,
                                            HATDecoderForCausalLM,
                                            HATEncoderForCausalLM)
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import weak_ref_tensor
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.hat.hat_utils import (COMPRESSION_RATIO, HATBatchInput,
                                   HATSubmodelRole)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class HATModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.role: Optional[HATSubmodelRole] = None

        self.use_cuda_graph = not self.model_config.enforce_eager
        # Tensors for CUDA Graph Caching, set in load_model
        self.previous_hidden_states: Optional[torch.Tensor] = None
        self.predictive_word_embeddings: Optional[torch.Tensor] = None

        self.graphs = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.

        self.attn_metadata_ex = None
        self.logits_indices_ex = None
        self.attention_cuda_graphs_ex = None
        self.num_scheduled_tokens_ex = None
        self.decoder_model_runner: Optional[HATModelRunner] = None

    def set_decoder_model_runner(
            self, decoder_model_runner: "HATModelRunner") -> None:
        self.decoder_model_runner = decoder_model_runner

    def load_model(self) -> None:
        super().load_model()
        self._set_role()

        match self.role:
            case HATSubmodelRole.DECODER:
                self.previous_hidden_states = torch.zeros(
                    self.vllm_config.compilation_config.max_capture_size,
                    self.model_config.hf_config.hidden_size,
                    dtype=self.dtype,
                    device=self.device)
                self.predictive_word_embeddings = torch.zeros(
                    self.vllm_config.compilation_config.max_capture_size,
                    self.model_config.hf_config.cross_attention_config.
                    hidden_size_kv,
                    dtype=self.dtype,
                    device=self.device)
            case HATSubmodelRole.BACKBONE:
                self.previous_hidden_states = torch.zeros(
                    self.vllm_config.compilation_config.max_capture_size,
                    self.model_config.hf_config.hidden_size,
                    dtype=self.dtype,
                    device=self.device)

    def register_request(self, new_req_data: NewRequestData) -> None:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=sampling_params,
            pooling_params=new_req_data.pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, Any], torch.Tensor, Optional[SpecDecodeMetadata]]:

        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        self.input_batch.req_id_to_index = {}
        self.input_batch._req_ids = []
        self.input_batch.greedy_reqs = set()
        self.input_batch.random_reqs = set()
        max_num_logprobs = -1

        row_idx = 0
        input_ids = []
        block_ids = []
        for sched_req in scheduler_output.scheduled_new_reqs:
            if self.role != HATSubmodelRole.DECODER:
                block_ids = sched_req.block_ids
                req_id = sched_req.req_id
                sampling_params = sched_req.sampling_params
                if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                self.requests[req_id] = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=sched_req.prompt_token_ids,
                    mm_inputs=sched_req.mm_inputs,
                    mm_positions=sched_req.mm_positions,
                    sampling_params=sched_req.sampling_params,
                    pooling_params=sched_req.pooling_params,
                    generator=generator,
                    block_ids=block_ids,
                    num_computed_tokens=sched_req.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=sched_req.lora_request,
                )

            state = self.requests[sched_req.req_id]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                sched_req.req_id]

            prompt_token_ids = state.prompt_token_ids
            self.input_batch.num_computed_tokens_cpu[row_idx] = 0
            if self.role != HATSubmodelRole.BACKBONE:
                input_ids.extend(sched_req.prompt_token_ids[:num_scheduled_tokens])

            sampling_params = state.sampling_params
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.input_batch.temperature_cpu[row_idx] = -1.0
                self.input_batch.greedy_reqs.add(sched_req.req_id)
            else:
                self.input_batch.temperature_cpu[
                    row_idx] = sampling_params.temperature
                self.input_batch.random_reqs.add(sched_req.req_id)

            if sampling_params.logprobs is not None:
                max_num_logprobs = max(max_num_logprobs, sampling_params.logprobs)

            self.input_batch._req_ids.append(sched_req.req_id)
            self.input_batch.block_table.add_row(sched_req.block_ids, row_idx)
            self.input_batch.req_id_to_index[sched_req.req_id] = row_idx
            row_idx += 1

        req_data = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(req_data.req_ids):
            state = self.requests[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            self.input_batch.num_computed_tokens_cpu[row_idx] = req_data.num_computed_tokens[idx]

            num_computed_tokens = req_data.num_computed_tokens[idx]
            if self.role != HATSubmodelRole.BACKBONE:
                if num_new_tokens == 1:
                    if num_computed_tokens >= state.num_prompt_tokens:
                        input_ids.append(state.output_token_ids[num_computed_tokens - state.num_prompt_tokens])
                    else:
                        input_ids.append(state.prompt_token_ids[num_computed_tokens]) 
                elif num_new_tokens > 1:
                    if state.num_prompt_tokens > num_computed_tokens:
                        offset_prompts = min(num_new_tokens, state.num_prompt_tokens - num_computed_tokens)
                        input_ids.extend(state.prompt_token_ids[num_computed_tokens:num_computed_tokens + offset_prompts])
                        offset_outputs = max(0, num_computed_tokens + num_new_tokens - state.num_prompt_tokens)
                        if offset_outputs > 0:
                            input_ids.extend(state.output_token_ids[:offset_outputs])
                    else:
                        start_pos = num_computed_tokens - state.num_prompt_tokens
                        input_ids.extend(state.output_token_ids[start_pos:start_pos + num_new_tokens])

            sampling_params = state.sampling_params
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.input_batch.temperature_cpu[row_idx] = -1.0
                self.input_batch.greedy_reqs.add(req_id)
            else:
                self.input_batch.temperature_cpu[
                    row_idx] = sampling_params.temperature
                self.input_batch.random_reqs.add(req_id)
                
            if sampling_params.logprobs is not None:
                max_num_logprobs = max(max_num_logprobs, sampling_params.logprobs)

            if self.role != HATSubmodelRole.DECODER:
                state.num_computed_tokens = req_data.num_computed_tokens[idx]

                if not req_data.resumed_from_preemption[idx]:
                    # Append the new blocks to the existing block IDs.
                    for i in range(len(self.kv_cache_config.kv_cache_groups)):
                        state.block_ids[i].extend(req_data.new_block_ids[idx][i])
                else:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    state.block_ids = req_data.new_block_ids[idx]

            self.input_batch._req_ids.append(req_id)
            self.input_batch.block_table.add_row(state.block_ids, row_idx)
            self.input_batch.req_id_to_index[req_id] = row_idx
            row_idx += 1

        self.input_batch.sampling_metadata = self.input_batch._make_sampling_metadata()
        self.input_batch.sampling_metadata.max_num_logprobs = max_num_logprobs if max_num_logprobs != -1 else None
        input_ids = torch.tensor(input_ids, dtype=torch.int64).pin_memory()

        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        if self.role != HATSubmodelRole.BACKBONE:
            # Copy the tensors to the GPU.
            assert input_ids.shape[0] == total_num_scheduled_tokens
            self.input_ids[:total_num_scheduled_tokens].copy_(input_ids,
                                                            non_blocking=True)

        # Common case (1D positions)
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
        )

        attn_metadata: dict[str, Any] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id],
                    kv_cache_group_spec.kv_cache_spec,
                    self.attn_metadata_builders[kv_cache_group_id],
                )

            attn_metadata_i = (builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            ))

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        attention_cuda_graphs = all(
            b.can_run_in_cudagraph(common_attn_metadata)
            for b in self.attn_metadata_builders)

        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        spec_decode_metadata = None

        if self.role == HATSubmodelRole.ENCODER:
            self.decoder_model_runner.attn_metadata_ex = attn_metadata_i
            self.decoder_model_runner.attention_cuda_graphs_ex = attention_cuda_graphs
            self.decoder_model_runner.logits_indices_ex = logits_indices
            self.decoder_model_runner.num_scheduled_tokens_ex = num_scheduled_tokens
            self.decoder_model_runner.requests = self.requests

        return attn_metadata, attention_cuda_graphs, logits_indices, spec_decode_metadata, num_scheduled_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        hat_batch_input: Optional[HATBatchInput] = None,
        prepare_inputs: bool = True,
    ) -> Union[ModelRunnerOutput, IntermediateTensors, torch.Tensor]:
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        if not prepare_inputs:
            attn_metadata = self.attn_metadata_ex
            attention_cuda_graphs = self.attention_cuda_graphs_ex
            logits_indices = self.logits_indices_ex
            num_scheduled_tokens_np = self.num_scheduled_tokens_ex
        else:
            attn_metadata, attention_cuda_graphs, logits_indices, _, num_scheduled_tokens_np = self._prepare_inputs(
                scheduler_output)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        decoder_cuda_graph = (not self.role == HATSubmodelRole.DECODER) or hat_batch_input.word_lens_bytes is None \
            or hat_batch_input.word_lens_bytes.shape[0] == num_scheduled_tokens
        use_cuda_graph = self.use_cuda_graph and decoder_cuda_graph and num_scheduled_tokens <= self.cudagraph_batch_sizes[
            -1]

        if use_cuda_graph:
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            self._copy_cuda_graph_inputs(hat_batch_input, num_scheduled_tokens)
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.vllm_config.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                from vllm.utils import round_up
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # Prepare the decoder inputs.

        input_ids = self.input_ids[:num_input_tokens]
        positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        model_kwargs = self._get_model_kwargs(input_ids, positions,
                                              num_input_tokens,
                                              hat_batch_input, use_cuda_graph)
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens,
                                 num_tokens_across_dp=num_tokens_across_dp,
                                 skip_cuda_graphs=skip_cuda_graphs):
            self.maybe_setup_kv_connector(scheduler_output)

            if use_cuda_graph:
                graph, model_output = self.graphs[num_input_tokens]
                graph.replay()
            else:
                model_output = self.model(**model_kwargs)

            self.maybe_wait_for_kv_save()

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output

        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0

        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        elif self.role == HATSubmodelRole.DECODER:
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        else:
            return hidden_states[:num_scheduled_tokens, :]

        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        # No spec decode tokens.
        valid_sampled_token_ids = sampled_token_ids.tolist()

        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        spec_token_ids = None

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            num_nans_in_logits=num_nans_in_logits,
        )

    def _copy_cuda_graph_inputs(self, hat_batch_input: HATBatchInput,
                                num_scheduled_tokens: int) -> None:
        match self.role:
            case HATSubmodelRole.DECODER:
                self.previous_hidden_states[:num_scheduled_tokens, :].copy_(
                    hat_batch_input.encoder_hidden_states, non_blocking=True)
                self.predictive_word_embeddings[:num_scheduled_tokens, :].copy_(
                    hat_batch_input.predictive_word_embeddings,
                    non_blocking=True)
            case HATSubmodelRole.BACKBONE:
                self.previous_hidden_states[:num_scheduled_tokens, :].copy_(
                    hat_batch_input.latent_word_embeddings, non_blocking=True)
            case HATSubmodelRole.ENCODER:
                pass

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_input_tensors: bool = False,
    ) -> torch.Tensor:

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            query_start_loc = self.query_start_loc[:num_reqs + 1]
            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=num_tokens,
            )

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                attn_metadata_i = self.attn_metadata_builders[
                    kv_cache_group_id].build_for_cudagraph_capture(
                        common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            model_kwargs = self._get_dummy_model_kwargs(
                num_tokens, create_tensors=create_input_tensors)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp):
                outputs = model(
                    **model_kwargs,
                    intermediate_tensors=intermediate_tensors,
                )

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return outputs[logit_indices]

    def profile_run(self) -> None:
        hidden_states = self._dummy_run(
            max(1, self.max_num_tokens // COMPRESSION_RATIO)
            if self.role == HATSubmodelRole.BACKBONE else self.max_num_tokens,
            create_input_tensors=True)
        if get_pp_group(
        ).is_last_rank and self.role == HATSubmodelRole.DECODER:
            sampler_output = self._dummy_sampler_run(hidden_states)
        else:
            sampler_output = None
        self._sync_device()
        del hidden_states, sampler_output
        gc.collect()

    @torch.inference_mode()
    def _capture(
        self,
        num_tokens: int,
        graph_capture_context: GraphCaptureContext,
    ) -> torch.Tensor:

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        # Make sure max_model_len is used at the graph capture time.
        self.seq_lens_np[:num_reqs] = self.max_model_len
        self.seq_lens_np[num_reqs:] = 0
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)
        seq_lens = self.seq_lens[:num_reqs]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc, 
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens
        )

        attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            attn_metadata_i = self.attn_metadata_builders[
                kv_cache_group_id].build_for_cudagraph_capture(
                    common_attn_metadata)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        model = self.model
        model_kwargs = self._get_dummy_model_kwargs(num_tokens,
                                                    create_tensors=False)

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device))

            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_tokens, None, False)

        torch.cuda.synchronize()
        # Capture the graph.
        graph = torch.cuda.CUDAGraph()
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_tokens,
                                 num_tokens_across_dp=num_tokens_across_dp):
            with torch.cuda.graph(graph,
                                  pool=self.graph_memory_pool,
                                  stream=graph_capture_context.stream):
                outputs = model(
                    **model_kwargs,
                    intermediate_tensors=intermediate_tensors,
                )
                if isinstance(outputs, torch.Tensor):
                    graph_outputs = weak_ref_tensor(outputs)
                elif isinstance(outputs, IntermediateTensors):
                    graph_outputs = IntermediateTensors(
                        tensors={
                            key: weak_ref_tensor(value)
                            for key, value in outputs.tensors.items()
                        })

                del outputs
                gc.collect()
            self.graph_memory_pool = graph.pool()
            self.graphs[num_tokens] = (graph, graph_outputs)

        torch.cuda.synchronize()
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return graph_outputs[logit_indices]

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device) as graph_capture_context:
            full_cg = self.full_cuda_graph
            # Only rank 0 should print progress bar during capture
            compilation_cases = reversed(self.cudagraph_batch_sizes)
            if is_global_first_rank():
                compilation_cases = tqdm(
                    list(compilation_cases),
                    disable=not self.load_config.use_tqdm_on_load,
                    desc="Capturing CUDA graph shapes")
            for num_tokens in compilation_cases:
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens, capture_attn_cudagraph=full_cg)
                self._capture(num_tokens, graph_capture_context=graph_capture_context)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def _get_dummy_model_kwargs(
            self,
            num_input_tokens: int,
            create_tensors: bool = False) -> dict[str, Any]:
        match self.role:
            case HATSubmodelRole.ENCODER:
                model_kwargs = {
                    "input_ids": self.input_ids[:num_input_tokens],
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                }
            case HATSubmodelRole.BACKBONE:
                previous_hidden_states = self.previous_hidden_states[:
                                                                     num_input_tokens, :]
                if create_tensors:
                    previous_hidden_states = torch.zeros(
                        num_input_tokens,
                        self.model_config.hf_config.hidden_size,
                        dtype=self.dtype,
                        device=self.device)
                model_kwargs = {
                    "input_ids": None,
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                    "previous_hidden_states": previous_hidden_states,
                }
            case HATSubmodelRole.DECODER:
                previous_hidden_states = self.previous_hidden_states[:
                                                                     num_input_tokens, :]
                predictive_word_embeddings = self.predictive_word_embeddings[:
                                                                             num_input_tokens, :]
                if create_tensors:
                    previous_hidden_states = torch.zeros(
                        num_input_tokens,
                        self.model_config.hf_config.hidden_size,
                        dtype=self.dtype,
                        device=self.device)
                    predictive_word_embeddings = torch.zeros(
                        num_input_tokens,
                        self.model_config.hf_config.cross_attention_config.
                        hidden_size_kv,
                        dtype=self.dtype,
                        device=self.device)
                model_kwargs = {
                    # Reuse input_ids for word positions, because the decoder does not need input_ids
                    "input_ids": None,
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                    "word_lens_bytes": None,
                    "predictive_word_embeddings": predictive_word_embeddings,
                    "previous_hidden_states": previous_hidden_states,
                }
        return model_kwargs

    def _get_model_kwargs(self,
                          input_ids: torch.Tensor,
                          positions: torch.Tensor,
                          num_input_tokens: int,
                          hat_batch_input: Optional[HATBatchInput] = None,
                          use_cuda_graph: bool = False) -> dict[str, Any]:
        match self.role:
            case HATSubmodelRole.ENCODER:
                model_kwargs = {
                    "input_ids": input_ids,
                    "inputs_embeds": None,
                    "positions": positions,
                }
            case HATSubmodelRole.BACKBONE:
                previous_hidden_states = hat_batch_input.latent_word_embeddings
                if use_cuda_graph:
                    previous_hidden_states = self.previous_hidden_states[:num_input_tokens, :]
                model_kwargs = {
                    "input_ids": None,
                    "inputs_embeds": None,
                    "positions": positions,
                    "previous_hidden_states": previous_hidden_states,
                }
            case HATSubmodelRole.DECODER:
                previous_hidden_states = hat_batch_input.encoder_hidden_states
                predictive_word_embeddings = hat_batch_input.predictive_word_embeddings
                word_lens_bytes = hat_batch_input.word_lens_bytes
                if use_cuda_graph:
                    previous_hidden_states = self.previous_hidden_states[:num_input_tokens, :]
                    predictive_word_embeddings = self.predictive_word_embeddings[:num_input_tokens, :]
                    word_lens_bytes = None
                model_kwargs = {
                    # Reuse input_ids for word positions, because the decoder does not need input_ids
                    "input_ids": None,
                    "inputs_embeds": None,
                    "positions": positions,
                    "word_lens_bytes": word_lens_bytes,
                    "predictive_word_embeddings": predictive_word_embeddings,
                    "previous_hidden_states": previous_hidden_states,
                }
        return model_kwargs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)

    def _set_role(self) -> None:
        if isinstance(self.model, HATEncoderForCausalLM):
            self.role = HATSubmodelRole.ENCODER
        elif isinstance(self.model, HATDecoderForCausalLM):
            self.role = HATSubmodelRole.DECODER
        elif isinstance(self.model, HATBackboneForCausalLM):
            self.role = HATSubmodelRole.BACKBONE
        else:
            raise ValueError(
                f"Unknown HAT model role: {type(self.model)}. Can't instantiate HATModelRunner."
            )
