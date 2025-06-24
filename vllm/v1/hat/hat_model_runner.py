# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc
import time
import weakref
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder)
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture,
    prepare_communication_buffer_for_model)
from vllm.forward_context import (DPMetadata, get_forward_context,
                                  set_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.hat import HATBackboneForCausalLM, HATDecoderForCausalLM, HATEncoderForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LazyLoader, async_tensor_h2d, cdiv,
                        check_use_alibi, is_pin_memory_available)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.hat.hat_utils import HATBatchInput, HATSubmodelRole
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class HATModelRunner(GPUModelRunner):

    def __init__( self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.role: Optional[HATSubmodelRole] = None

        # Tensors for CUDA Graph Caching, set in load_model
        
        self.previous_hidden_states: Optional[torch.Tensor] = None
        self.predictive_word_embeddings: Optional[torch.Tensor] = None
        self.decoder_word_positions: Optional[torch.Tensor] = None

        self.scheduler_metadata = torch.zeros(self.max_num_reqs + 1,
                                              dtype=torch.int32,
                                              device=self.device)
    
    def load_model(self) -> None:
        super().load_model()
        self._set_role()

        match self.role:
            case HATSubmodelRole.DECODER:
                self.previous_hidden_states = torch.zeros(self.max_num_tokens,
                    self.model_config.hf_config.hidden_size,
                    dtype=self.dtype,
                    device=self.device)
                self.predictive_word_embeddings = torch.zeros(self.max_num_tokens,
                    self.model_config.hf_config.cross_attention_config.hidden_size_kv,
                    dtype=self.dtype,
                    device=self.device)
                self.decoder_word_positions = torch.zeros(
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device)
            case HATSubmodelRole.BACKBONE:
                self.previous_hidden_states = torch.zeros(self.max_num_tokens,
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
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the create new input batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # L TODO: Check how slow this is later
        req_ids = self.input_batch.req_ids[:]
        for id_ in req_ids:
            self.input_batch.remove_request(id_)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
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
                generator=generator,
                block_ids=copy.deepcopy(new_req_data.block_ids),
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for i in range(len(self.kv_cache_config.kv_cache_groups)):
                    req_state.block_ids[i].extend(copy.deepcopy(req_data.new_block_ids[i]))
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_ids_to_add.append(req_id)

        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)
        # We just want to execute the 2 trim statements at the end of the function
        self.input_batch.condense([])

        batch_reordered = self._may_reorder_batch(scheduler_output)
        assert not batch_reordered

        self.input_batch.refresh_sampling_metadata()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        hat_batch_input: Optional[HATBatchInput] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors, torch.Tensor]:

        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        attn_metadata, logits_indices, _ = (
            self._prepare_inputs(scheduler_output))

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        decoder_cuda_graph = (not self.role == HATSubmodelRole.DECODER) or (num_scheduled_tokens == hat_batch_input.word_positions.shape[0])
        use_cuda_graph = self.use_cuda_graph and decoder_cuda_graph and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
        use_cuda_graph = use_cuda_graph and num_scheduled_tokens == 1

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

        model_kwargs = self._get_model_kwargs(input_ids, positions, num_input_tokens, hat_batch_input, use_cuda_graph)
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens,
                                 num_tokens_across_dp=num_tokens_across_dp):
            self.maybe_setup_kv_connector(scheduler_output)

            model_output = self.model(**model_kwargs)

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

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

        spec_token_ids = None

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )
        
    def _copy_cuda_graph_inputs(self, hat_batch_input: HATBatchInput, num_scheduled_tokens: int) -> None:
        match self.role:
            case HATSubmodelRole.DECODER:
                self.previous_hidden_states[:num_scheduled_tokens, :].copy_(hat_batch_input.encoder_hidden_states, non_blocking=True)
                self.predictive_word_embeddings[:num_scheduled_tokens, :].copy_(hat_batch_input.predictive_word_embeddings, non_blocking=True)
                self.decoder_word_positions[:num_scheduled_tokens].copy_(hat_batch_input.word_positions, non_blocking=True)
            case HATSubmodelRole.BACKBONE:
                self.previous_hidden_states[:num_scheduled_tokens, :].copy_(hat_batch_input.latent_word_embeddings, non_blocking=True)
            case HATSubmodelRole.ENCODER:
                pass

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        skip_attn: bool = True,
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

        if skip_attn:
            attn_metadata: Optional[dict[str, Any]] = None
        else:
            query_start_loc = self.query_start_loc[:num_reqs + 1]
            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc, seq_lens=seq_lens)

            attn_metadata = {}
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                attn_metadata_i = (
                    self.attn_metadata_builders[kv_cache_group_id].build(
                        num_reqs=num_reqs,
                        num_actual_tokens=num_tokens,
                        max_query_len=num_tokens,
                        common_prefix_len=0,
                        common_attn_metadata=common_attn_metadata,
                    ))
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            model_kwargs = self._get_dummy_model_kwargs(num_tokens)

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

    def _get_dummy_model_kwargs(self, num_input_tokens: int) -> dict[str, Any]:
        match self.role:
            case HATSubmodelRole.ENCODER:
                model_kwargs = {
                    "input_ids": self.input_ids[:num_input_tokens],
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                }
            case HATSubmodelRole.BACKBONE:
                model_kwargs = {
                    "input_ids": None,
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                    "previous_hidden_states": self.previous_hidden_states[:num_input_tokens, :],
                }
            case HATSubmodelRole.DECODER:
                model_kwargs = {
                    # Reuse input_ids for word positions, because the decoder does not need input_ids
                    "input_ids": self.decoder_word_positions[:num_input_tokens],
                    "inputs_embeds": None,
                    "positions": self.positions[:num_input_tokens],
                    "word_len_bytes": None,
                    "predictive_word_embeddings": self.predictive_word_embeddings[:num_input_tokens, :],
                    "previous_hidden_states": self.previous_hidden_states[:num_input_tokens, :],
                }
        return model_kwargs

    def _get_model_kwargs(self, input_ids: torch.Tensor,
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
                word_positions = hat_batch_input.word_positions
                word_len_bytes = hat_batch_input.word_len_bytes
                if use_cuda_graph:
                    previous_hidden_states = self.previous_hidden_states[:num_input_tokens, :]
                    predictive_word_embeddings = self.predictive_word_embeddings[:num_input_tokens, :]
                    word_positions = self.decoder_word_positions[:num_input_tokens]
                    word_len_bytes = None

                model_kwargs = {
                    # Reuse input_ids for word positions, because the decoder does not need input_ids
                    "input_ids": word_positions,
                    "inputs_embeds": None,
                    "positions": positions,
                    "word_len_bytes": word_len_bytes,
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
            raise ValueError(f"Unknown HAT model role: {type(self.model)}. Can't instantiate HATModelRunner.")