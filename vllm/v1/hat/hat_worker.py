# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import dataclasses
import gc
import os
from typing import List, Optional

import torch
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             init_model_parallel_group,
                                             patch_tensor_parallel_group)
from vllm.logger import init_logger
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname, MemorySnapshot
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hat.hat_manager import HATManager
from vllm.v1.hat.hat_model_runner import HATModelRunner
from vllm.v1.hat.hat_utils import (BYTES_PER_WORKER_STEP,
                                   LIMIT_FOR_STATIC_STEPS, HATBatchInput,
                                   _allocate_kv_cache_tensors,
                                   _reshape_kv_cache_tensors, safe_list_slice)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, _check_if_gpu_supports_dtype, init_worker_distributed_environment
from vllm.v1.worker.utils import bind_kv_cache
from vllm.v1.worker.worker_base import WorkerBase


logger = init_logger(__name__)


def create_hat_worker(*args, **kwargs):
    ModelRegistry.register_model(
        "HATEncoderForCausalLM",
        "vllm.model_executor.models.hat:HATEncoderForCausalLM")
    ModelRegistry.register_model(
        "HATBackboneForCausalLM",
        "vllm.model_executor.models.hat:HATBackboneForCausalLM")

    vllm_config: VllmConfig = kwargs.get("vllm_config")
    vllm_config.model_config.disable_cascade_attn = True
    encoder_worker = _create_worker(
        vllm_config.model_config.hf_config.encoder_config,
        "HATEncoderForCausalLM", *args, **kwargs)
    decoder_worker = _create_worker(
        vllm_config.model_config.hf_config.decoder_config,
        "HATDecoderForCausalLM", *args, **kwargs)
    backbone_worker = _create_worker(
        vllm_config.model_config.hf_config.backbone_config,
        "HATBackboneForCausalLM", *args, **kwargs)
    return HATWorker(encoder_worker, decoder_worker, backbone_worker,
                     vllm_config)


def _create_worker(hf_config: PretrainedConfig, architecture: str, *args,
                   **kwargs):
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    worker_config = copy.deepcopy(vllm_config)
    hf_config.architectures = [architecture]
    worker_config.model_config.hf_config = hf_config
    worker_config.model_config.hf_text_config = hf_config

    if worker_config.model_config.hf_config.sliding_window is None:
        worker_config.cache_config.sliding_window = None
    if architecture != "HATBackboneForCausalLM":
        worker_config.parallel_config.tensor_parallel_size = 1
        worker_config.parallel_config.world_size = 1

    kwargs["vllm_config"] = worker_config
    kwargs["model_runner_cls"] = HATModelRunner
    worker_class = resolve_obj_by_qualname(
        "vllm.v1.hat.hat_worker.HATModelWorker")
    worker = worker_class(*args, **kwargs)
    return worker


class HATWorker(WorkerBase):

    def __init__(self, encoder_worker: WorkerBase, decoder_worker: WorkerBase,
                 backbone_worker: WorkerBase, vllm_config: VllmConfig):
        self.encoder_worker = SmallerTpWorker(
            encoder_worker
        ) if vllm_config.parallel_config.tensor_parallel_size > 1 else encoder_worker
        self.decoder_worker = SmallerTpWorker(
            decoder_worker
        ) if vllm_config.parallel_config.tensor_parallel_size > 1 else decoder_worker
        self.backbone_worker = backbone_worker

        self.encoder_connector = None
        self.hat_manager = None

        self.vllm_config = vllm_config

        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        pass        

    def init_device(self) -> None:
        self.backbone_worker.init_device()
        self.backbone_worker.load_model()
        self.encoder_worker.init_device()
        self.encoder_worker.load_model()
        self.decoder_worker.init_device()
        self.decoder_worker.load_model()

        self.encoder_model_runner = self.encoder_worker.model_runner
        self.decoder_model_runner = self.decoder_worker.model_runner
        self.backbone_model_runner = self.backbone_worker.model_runner

        # L TODO: This is a hack to get the decoder to use the same input batch as the encoder. Refactor this by putting both
        # into the same model runner.
        self.decoder_model_runner.input_batch = self.encoder_model_runner.input_batch
        self.decoder_model_runner.input_ids = self.encoder_model_runner.input_ids
        self.decoder_model_runner.positions = self.encoder_model_runner.positions
        self.decoder_model_runner.seq_lens = self.encoder_model_runner.seq_lens
        self.decoder_model_runner.slot_mapping = self.encoder_model_runner.slot_mapping
        self.decoder_model_runner.query_start_loc = self.encoder_model_runner.query_start_loc
        self.encoder_model_runner.set_decoder_model_runner(
            self.decoder_model_runner)

        self.default_stream = torch.cuda.default_stream()
        self.stream_backbone = torch.cuda.Stream()
        self.stream_enc_dec = torch.cuda.Stream()

    def load_model(self) -> None:
        pass

    def determine_available_memory(self) -> int:
        available_memory = self.backbone_worker.determine_available_memory()
        return available_memory

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        encoder_spec = self.encoder_worker.get_kv_cache_spec()
        sliding_window_spec_example = encoder_spec[next(
            iter(encoder_spec.keys()))]
        decoder_spec = self.decoder_worker.get_kv_cache_spec()
        backbone_spec = self.backbone_worker.get_kv_cache_spec()
        full_attention_spec_example = backbone_spec[next(
            iter(backbone_spec.keys()))]

        assert sliding_window_spec_example.page_size_bytes % full_attention_spec_example.page_size_bytes == 0, "It is not possible to handle this KV cache configuration"
        ratio = sliding_window_spec_example.page_size_bytes // full_attention_spec_example.page_size_bytes
        for layer_spec in backbone_spec.values():
            layer_spec.block_size *= ratio

        encoder_spec.update(decoder_spec)
        encoder_spec.update(backbone_spec)

        return encoder_spec

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # Here we call self.model_runner.initialize_kv_cache
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            kv_cache_config_backbone = KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_config.kv_cache_groups[:-1])
            kv_cache_config_enc_dec = KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_config.kv_cache_groups[-1:])
            self.decoder_model_runner.initialize_kv_cache(
                kv_cache_config_enc_dec)
            self.backbone_model_runner.initialize_kv_cache(
                kv_cache_config_backbone)
            self.encoder_model_runner.initialize_kv_cache(
                kv_cache_config_enc_dec)

        # Now the model runner has the attention backends initialised
        # 1. Create kv_caches
        # 2. Bind relevant KV_caches to each model runner

        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = _allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        attn_backends = [
            *self.backbone_model_runner.attn_backends,
            self.encoder_model_runner.attn_backends[0]
        ]

        kv_caches = _reshape_kv_cache_tensors(kv_cache_config,
                                              kv_cache_raw_tensors,
                                              attn_backends)

        # We want to bind the kv_cache separately for each model runner
        encoder_forward_context = self.encoder_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_enc = {
            k: v
            for k, v in kv_caches.items() if k in encoder_forward_context
        }
        bind_kv_cache(filtered_kv_caches_enc, encoder_forward_context,
                      self.encoder_model_runner.kv_caches)

        decoder_forward_context = self.decoder_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_dec = {
            k: v
            for k, v in kv_caches.items() if k in decoder_forward_context
        }
        bind_kv_cache(filtered_kv_caches_dec, decoder_forward_context,
                      self.decoder_model_runner.kv_caches)

        backbone_forward_context = self.backbone_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_backbone = {
            k: v
            for k, v in kv_caches.items() if k in backbone_forward_context
        }
        bind_kv_cache(filtered_kv_caches_backbone, backbone_forward_context,
                      self.backbone_model_runner.kv_caches)

        self.hat_manager = HATManager(
            vllm_config=self.vllm_config,
            backbone_model_runner=self.backbone_model_runner,
            device=self.device,
            rank=self.rank,
            driver_rank=self.driver_rank)
        self._setup_modules()

    def compile_or_warm_up_model(self) -> None:
        self.encoder_worker.compile_or_warm_up_model()
        self.backbone_worker.compile_or_warm_up_model()
        self.decoder_worker.compile_or_warm_up_model(True)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        self.hat_manager.remove_finished_requests(scheduler_output)
        if not scheduler_output.num_scheduled_tokens:
            return self._handle_empty_scheduler_output(scheduler_output)

        self.default_stream.wait_stream(self.stream_backbone)
        self.default_stream.wait_stream(self.stream_enc_dec)
        scheduler_output_byte, scheduler_output_word = self.hat_manager.add_request(
            scheduler_output)
        encoder_hidden_states = self.encoder_worker.execute_model(
            scheduler_output_byte)

        encoder_hidden_states_phases, scheduler_output_byte_enc_dec, scheduler_output_byte_final_decoder = (
            self.hat_manager.handle_encoder_output(scheduler_output_byte,
                                                   encoder_hidden_states))

        predictive_word_embeddings = None
        if self._has_scheduled_requests(scheduler_output_word):
            self.stream_backbone.wait_stream(self.default_stream)
            self.stream_backbone.wait_stream(self.stream_enc_dec)
            with torch.cuda.stream(self.stream_backbone):
                predictive_word_embeddings = self.run_backbone(
                    scheduler_output_word, encoder_hidden_states_phases.
                    encoder_hidden_states_encoder_connector)
                self.hat_manager.update_backbone_info_prefill_path(
                    scheduler_output_word)
        
        if self._has_scheduled_requests(scheduler_output_byte_final_decoder):
            with torch.cuda.stream(self.stream_backbone):
                predictive_word_embeddings_final_decoder = self.hat_manager.handle_backbone_output(
                    scheduler_output_byte_final_decoder,
                    predictive_word_embeddings)

        both_paths_active = self._has_scheduled_requests(
            scheduler_output_byte_enc_dec) and self._has_scheduled_requests(
                scheduler_output_byte_final_decoder)
        if self._has_scheduled_requests(scheduler_output_byte_enc_dec):
            self.stream_enc_dec.wait_stream(self.default_stream)
            with torch.cuda.stream(self.stream_enc_dec):
                self.run_decode_loop(encoder_hidden_states_phases.
                                     encoder_hidden_states_enc_dec_loop,
                                     scheduler_output_byte_enc_dec,
                                     prepare_inputs=True)

        if self._has_scheduled_requests(scheduler_output_byte_final_decoder):
            with torch.cuda.stream(self.stream_backbone):
                self.stream_backbone.wait_stream(self.default_stream)
                self.stream_backbone.wait_stream(self.stream_enc_dec)
                word_lens_bytes = self.hat_manager.prepare_input_final_decoder(
                    scheduler_output_byte_final_decoder)

                hat_batch_input_final_decoder = HATBatchInput(
                    predictive_word_embeddings=
                    predictive_word_embeddings_final_decoder,
                    encoder_hidden_states=encoder_hidden_states_phases.
                    encoder_hidden_states_final_decoder,
                    word_lens_bytes=word_lens_bytes)
                model_runner_output = self.decoder_worker.execute_model(
                    scheduler_output_byte_final_decoder,
                    hat_batch_input_final_decoder,
                    prepare_inputs=both_paths_active)

                scheduled_cached_reqs_dec_word_boundary_req_ids = safe_list_slice(
                    scheduler_output_byte_final_decoder.scheduled_cached_reqs.req_ids,
                    self.hat_manager.num_decodes_word_boundary,
                    keep_prefix=False)
                self.hat_manager.process_outputs_enc_dec_loop(
                    scheduled_cached_reqs_dec_word_boundary_req_ids,
                    model_runner_output,
                    decode_path=False)
                self.hat_manager.process_outputs_prefill_chunked_prefill(
                    scheduler_output_byte_final_decoder, model_runner_output)

        scheduler_output_word_decodes = self.hat_manager.scheduler_output_word_decodes
        if self._has_scheduled_requests(scheduler_output_word_decodes):
            self.stream_enc_dec.wait_stream(self.stream_backbone)
            with torch.cuda.stream(self.stream_enc_dec):
                predictive_word_embeddings = self.run_backbone(
                    scheduler_output_word_decodes)
                self.hat_manager.update_backbone_info_decode_path(
                    predictive_word_embeddings)
        
        return self.hat_manager.finish_step()

    def run_backbone(
        self,
        scheduler_output: SchedulerOutput,
        encoder_hidden_states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        encoder_connector_input = self.hat_manager.prepare_input_encoder_connector(
            scheduler_output, encoder_hidden_states)
        updated_latent_word_embeddings = self.encoder_connector(
            **dataclasses.asdict(encoder_connector_input))
        hat_batch_input = HATBatchInput(
            latent_word_embeddings=updated_latent_word_embeddings)
        predictive_word_embeddings = self.backbone_worker.execute_model(
            scheduler_output, hat_batch_input)
        return predictive_word_embeddings

    def run_decode_loop(self,
                        encoder_hidden_states: torch.Tensor,
                        scheduler_output: SchedulerOutput,
                        prepare_inputs: bool = True):
        predictive_word_embeddings = self.hat_manager.prepare_exec_model_req_for_dec_autoregressive_phase(
            scheduler_output)
        hat_batch_input = HATBatchInput(
            predictive_word_embeddings=predictive_word_embeddings,
            encoder_hidden_states=encoder_hidden_states)
        model_runner_output = self.decoder_worker.execute_model(
            scheduler_output, hat_batch_input, prepare_inputs=prepare_inputs)
        scheduler_output = self.hat_manager.process_outputs_enc_dec_loop(
            scheduler_output.scheduled_cached_reqs.req_ids, model_runner_output)
        num_decodes_running = len(scheduler_output.scheduled_cached_reqs.req_ids)

        process_full_word = num_decodes_running <= LIMIT_FOR_STATIC_STEPS

        bytes_processed = 0
        while num_decodes_running > 0 and (
                bytes_processed < BYTES_PER_WORKER_STEP or process_full_word):
            encoder_hidden_states = self.encoder_worker.execute_model(
                scheduler_output)
            self.hat_manager.handle_encoder_output_loop(
                encoder_hidden_states, scheduler_output)

            predictive_word_embeddings = self.hat_manager.prepare_exec_model_req_for_dec_autoregressive_phase(
                scheduler_output)

            hat_batch_input = HATBatchInput(
                predictive_word_embeddings=predictive_word_embeddings,
                encoder_hidden_states=encoder_hidden_states)
            model_runner_output = self.decoder_worker.execute_model(
                scheduler_output, hat_batch_input, prepare_inputs=False)
            scheduler_output = self.hat_manager.process_outputs_enc_dec_loop(
                scheduler_output.scheduled_cached_reqs.req_ids, model_runner_output)
            bytes_processed += 1
            num_decodes_running = len(scheduler_output.scheduled_cached_reqs.req_ids)

    def _handle_empty_scheduler_output(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        self.encoder_worker.execute_model(scheduler_output)
        self.decoder_worker.execute_model(scheduler_output)
        self.backbone_worker.execute_model(scheduler_output)
        return EMPTY_MODEL_RUNNER_OUTPUT

    def _setup_modules(self):
        self.encoder_connector = self.encoder_worker.get_model(
        ).encoder_connector
        self.hat_manager.first_word_embedding = (
            self.backbone_worker.get_model(
            ).decoder_connector.first_word_embedding.squeeze(0))

    def _has_scheduled_requests(self,
                                scheduler_output: SchedulerOutput) -> bool:
        return bool(scheduler_output.scheduled_new_reqs
                    or scheduler_output.scheduled_cached_reqs.req_ids)

    @property
    def rank(self):
        return self.backbone_worker.rank

    @property
    def device(self):
        return self.backbone_worker.device

    @property
    def driver_rank(self) -> int:
        return 0

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))


class HATModelWorker(Worker):

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        hat_batch_input: Optional[HATBatchInput] = None,
        prepare_inputs: bool = True,
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors,
                                                 hat_batch_input,
                                                 prepare_inputs)
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        return output

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            # Maybe add a memory check as done in the gpu worker specialized for HAT
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        if self.model_runner_cls is None:
            self.model_runner = GPUModelRunner(
                self.vllm_config, self.device)
        else:
            self.model_runner = self.model_runner_cls(
                self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def compile_or_warm_up_model(self, run_sampler: bool = False) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.model_runner.vllm_config.compilation_config.compile_sizes.copy(
        )
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, create_input_tensors=True)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank and run_sampler:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            self.model_runner._dummy_sampler_run(
                hidden_states=self.model_runner._dummy_run(
                    num_tokens=max_num_reqs, create_input_tensors=True))

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)


class SmallerTpWorker:
    """Class which allows us to run the smaller byte models replicated across
    the TP ranks.
    """

    def __init__(self, worker: HATModelWorker):
        self._worker = worker
        self._tp_group = None
        self._patched_methods = [
            "load_model",
            "determine_available_memory",
            "get_kv_cache_spec",
            "initialize_from_config",
            "compile_or_warm_up_model",
            "get_model",
            "execute_model",
        ]

    def _patch_tensor_parallel_group(self):
        """Temporarily patch the global tp group state with its own tp group
        state.
        """
        return patch_tensor_parallel_group(self._tp_group)

    def init_device(self) -> None:
        # creates tp process group containing only a subset of gpu ranks
        local_rank = get_tp_group().local_rank
        tp_backend = torch.distributed.get_backend(get_tp_group().device_group)
        self._tp_group = init_model_parallel_group([[local_rank]], local_rank,
                                                   tp_backend)

        with self._patch_tensor_parallel_group():
            self._worker.init_device()

    def __getattr__(self, name):
        attr = getattr(self._worker, name)
        if name in self._patched_methods and callable(attr):

            def wrapped(*args, **kwargs):
                with self._patch_tensor_parallel_group():
                    return attr(*args, **kwargs)

            return wrapped
        return attr
