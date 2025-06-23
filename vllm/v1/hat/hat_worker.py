from typing import List, Optional

import torch
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.utils import set_random_seed
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hat.hat_manager import HATManager
from vllm.v1.hat.hat_model_runner import HATModelRunner
from vllm.v1.hat.hat_utils import HATBatchInput, safe_list_slice
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerBase
from transformers import PretrainedConfig
import copy

logger = init_logger(__name__)


def create_hat_worker(*args, **kwargs):
    ModelRegistry.register_model(
        "HATEncoderForCausalLM", 
        "vllm.model_executor.models.hat:HATEncoderForCausalLM"
    )
    ModelRegistry.register_model(
        "HATBackboneForCausalLM", 
        "vllm.model_executor.models.hat:HATBackboneForCausalLM"
    )
        
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    encoder_worker = _create_worker(vllm_config.model_config.hf_config.encoder_config,
                                    "HATEncoderForCausalLM",
                                    *args, **kwargs)
    decoder_worker = _create_worker(vllm_config.model_config.hf_config.decoder_config,
                                    "HATDecoderForCausalLM",
                                    *args, **kwargs)
    backbone_worker = _create_worker(vllm_config.model_config.hf_config.backbone_config,
                                     "HATBackboneForCausalLM",
                                     *args, **kwargs)
    return HATWorker(encoder_worker, decoder_worker, backbone_worker, vllm_config)


def _create_worker(hf_config: PretrainedConfig, architecture: str, *args, **kwargs):
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    worker_config = copy.deepcopy(vllm_config)
    hf_config.architectures = [architecture]
    worker_config.model_config.hf_config = hf_config
    worker_config.model_config.hf_text_config = hf_config

    if worker_config.model_config.hf_config.sliding_window is None:
        worker_config.cache_config.sliding_window = None

    kwargs["vllm_config"] = worker_config
    kwargs["model_runner_cls"] = HATModelRunner
    worker_class = resolve_obj_by_qualname("vllm.v1.hat.hat_worker.HATModelWorker")
    worker = worker_class(*args, **kwargs)
    return worker


class HATWorker(WorkerBase):

    def __init__(self, encoder_worker: WorkerBase,
                 decoder_worker: WorkerBase,
                 backbone_worker: WorkerBase,
                 vllm_config: VllmConfig):
        self.encoder_worker = encoder_worker
        self.decoder_worker = decoder_worker
        self.backbone_worker = backbone_worker
        
        self.encoder_connector = None
        self.use_cuda_graph = vllm_config.compilation_config.max_capture_size > 0
        self.hat_manager = None
        self.pred_word_embeds_buffer = None
        self.backbone_num_gpu_blocks: Optional[int] = None
        self.process_full_word = False
        
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

        #self.text = "An Apple a day keeps the doctor away"
        self.text = "🚀🎉🔥⭐".encode("utf-8")
        self.idx = 0
        self.steps = 0
        
    def init_device(self) -> None:
        self.encoder_worker.init_device()
        self.encoder_worker.load_model()
        self.decoder_worker.init_device()
        self.decoder_worker.load_model()
        self.backbone_worker.init_device()
        self.backbone_worker.load_model()
        
        self.encoder_model_runner = self.encoder_worker.model_runner
        self.decoder_model_runner = self.decoder_worker.model_runner
        self.backbone_model_runner = self.backbone_worker.model_runner
        self.first = True
    
    def load_model(self) -> None:
        pass
    
    def determine_available_memory(self) -> int:
        # TODO
        return int(10e9)
    
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        encoder_spec = self.encoder_worker.get_kv_cache_spec()
        decoder_spec = self.decoder_worker.get_kv_cache_spec()
        backbone_spec = self.backbone_worker.get_kv_cache_spec()
        
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
                kv_cache_groups=kv_cache_config.kv_cache_groups[:-1]
            )
            kv_cache_config_enc_dec = KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_config.kv_cache_groups[-1:]
            )
            self.decoder_model_runner.initialize_kv_cache(kv_cache_config_enc_dec)
            self.backbone_model_runner.initialize_kv_cache(kv_cache_config_backbone)
            self.encoder_model_runner.initialize_kv_cache(kv_cache_config_enc_dec)
                    
        # Now the model runner has the attention backends initialised
        # 1. Create kv_caches
        # 2. Bind relevant KV_caches to each model runner
        
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = _allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        attn_backends = [*self.backbone_model_runner.attn_backends, self.encoder_model_runner.attn_backends[0]]
    
        kv_caches = _reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors, attn_backends)
        
        # We want to bind the kv_cache separately for each model runner
        encoder_forward_context = self.encoder_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_enc = {k: v for k, v in kv_caches.items() if k in encoder_forward_context}
        bind_kv_cache(filtered_kv_caches_enc, encoder_forward_context, self.encoder_model_runner.kv_caches)
        
        decoder_forward_context = self.decoder_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_dec = {k: v for k, v in kv_caches.items() if k in decoder_forward_context}
        bind_kv_cache(filtered_kv_caches_dec, decoder_forward_context, self.decoder_model_runner.kv_caches)
        
        backbone_forward_context = self.backbone_model_runner.vllm_config.compilation_config.static_forward_context
        filtered_kv_caches_backbone = {k: v for k, v in kv_caches.items() if k in backbone_forward_context}
        bind_kv_cache(filtered_kv_caches_backbone, backbone_forward_context, self.backbone_model_runner.kv_caches)

        self.hat_manager = HATManager(special_token_dict=self.vllm_config.model_config.hf_config.special_token_dict,
                                      max_word_size = self.vllm_config.model_config.hf_config.max_word_size,
                                      backbone_model_runner=self.backbone_model_runner,
                                      device=self.device,
                                      rank=self.rank,
                                      driver_rank=self.driver_rank)
        self._setup_modules()
        
    def _compile_or_warm_up_model(self, model_runner, run_sampler: bool = False) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = model_runner.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank and run_sampler:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            model_runner._dummy_sampler_run(
                hidden_states=model_runner._dummy_run(
                    num_tokens=max_num_reqs))

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        
    def compile_or_warm_up_model(self) -> None:
        self._compile_or_warm_up_model(self.backbone_model_runner)
        self._compile_or_warm_up_model(self.encoder_model_runner)
        self._compile_or_warm_up_model(self.decoder_model_runner, True)    

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:

        if not scheduler_output.num_scheduled_tokens:
            return self._handle_empty_scheduler_output(scheduler_output)
        
        scheduler_output_byte, scheduler_output_word = self.hat_manager.add_request(scheduler_output)
        encoder_hidden_states = self.encoder_worker.execute_model(scheduler_output_byte)

        encoder_hidden_states_encoder_connector, encoder_hidden_states_enc_dec_loop, encoder_hidden_states_final_decoder, scheduler_output_byte_enc_dec, scheduler_output_byte_final_decoder = (
            self.hat_manager.handle_encoder_output(scheduler_output_byte, encoder_hidden_states)
        )

        if encoder_hidden_states_enc_dec_loop is not None:
            self.run_decode_loop(encoder_hidden_states_enc_dec_loop, scheduler_output_byte_enc_dec)
        
        if len(scheduler_output_byte_final_decoder.scheduled_new_reqs) == 0 and len(scheduler_output_byte_final_decoder.scheduled_cached_reqs) == 0:
            output = self.hat_manager.output
            self.hat_manager.reset_manager()
            return output

        predictive_word_embeddings = None
        if len(scheduler_output_word.scheduled_new_reqs) > 0 or len(scheduler_output_word.scheduled_cached_reqs) > 0:
            encoder_hidden_states_encoder_connector, word_lens_bytes_per_task_excl_last_word, byte_positions, word_positions = (
                self.hat_manager.prepare_input_encoder_connector(encoder_hidden_states_encoder_connector, scheduler_output_word)
            )

            updated_latent_word_embeddings = self.encoder_connector(encoder_hidden_states_encoder_connector,
                                                                    byte_positions,
                                                                    word_positions,
                                                                    word_lens_bytes_per_task_excl_last_word)
            
            hat_batch_input = HATBatchInput(latent_word_embeddings=updated_latent_word_embeddings)
            predictive_word_embeddings = self.backbone_worker.execute_model(scheduler_output_word, hat_batch_input)

        predictive_word_embeddings_final_decoder = self.hat_manager.handle_backbone_output(scheduler_output_byte_final_decoder, predictive_word_embeddings)
        self.hat_manager.update_backbone_info(scheduler_output_word)
        
        word_positions_final_decoder, word_len_bytes = self.hat_manager.prepare_input_final_decoder(scheduler_output_byte_final_decoder)
        
        if self.steps == 2:
            #exit()
            pass
        self.steps += 1
        
        hat_batch_input_final_decoder = HATBatchInput(predictive_word_embeddings=predictive_word_embeddings_final_decoder,
                                                      word_positions=word_positions_final_decoder,
                                                      encoder_hidden_states=encoder_hidden_states_final_decoder,
                                                      word_len_bytes=word_len_bytes)
        model_runner_output = self.decoder_worker.execute_model(scheduler_output_byte_final_decoder, hat_batch_input_final_decoder)
        
        scheduled_cached_reqs_dec_word_boundary = safe_list_slice(scheduler_output_byte_final_decoder.scheduled_cached_reqs,
                                                                  self.hat_manager.num_decodes_word_boundary,
                                                                  keep_prefix=False)
        self.hat_manager.process_outputs_enc_dec_loop(scheduled_cached_reqs_dec_word_boundary, model_runner_output)
        self.hat_manager.process_outputs_prefill_chunked_prefill(scheduler_output_byte_final_decoder, model_runner_output)

        output = self.hat_manager.output
        self.hat_manager.reset_manager()
        return output

        ids = []
        for req in scheduler_output.scheduled_new_reqs:
            ids.append(req.req_id)
        for req in scheduler_output.scheduled_cached_reqs:
            ids.append(req.req_id)
        
        sampled_token_ids = []
        for _ in ids:
            samples = []
            for _ in range(1):
                samples.append(98)
                #sampled_token_ids.append(self.text[self.idx])
                self.idx = (self.idx + 1) % len(self.text)
            sampled_token_ids.append(samples)

        output = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={id_: i for i, id_ in enumerate(ids)},
            sampled_token_ids=sampled_token_ids,
            spec_token_ids=None,
            prompt_logprobs_dict={id_: None for id_ in ids},
            logprobs=None,
        )

        return output
    
    def run_decode_loop(self, encoder_hidden_states: torch.Tensor, scheduler_output: SchedulerOutput):
        
        predictive_word_embeddings = self.hat_manager.prepare_exec_model_req_for_dec_autoregressive_phase(scheduler_output)
        word_positions = self.hat_manager.compute_position_ids_decoder_autoregressive_phase(scheduler_output)
        
        hat_batch_input = HATBatchInput(predictive_word_embeddings=predictive_word_embeddings,
                                        word_positions=word_positions,
                                        encoder_hidden_states=encoder_hidden_states)
        model_runner_output = self.decoder_worker.execute_model(scheduler_output, hat_batch_input)
        scheduler_output = self.hat_manager.process_outputs_enc_dec_loop(scheduler_output.scheduled_cached_reqs, model_runner_output)
        
        while self.hat_manager.num_decodes_not_word_boundary > 0:
            encoder_hidden_states = self.encoder_worker.execute_model(scheduler_output)
            self.hat_manager.handle_encoder_output_loop(encoder_hidden_states, scheduler_output)
            
            predictive_word_embeddings = self.hat_manager.prepare_exec_model_req_for_dec_autoregressive_phase(scheduler_output)
            word_positions = self.hat_manager.compute_position_ids_decoder_autoregressive_phase(scheduler_output)
            
            hat_batch_input = HATBatchInput(predictive_word_embeddings=predictive_word_embeddings,
                                            word_positions=word_positions,
                                            encoder_hidden_states=encoder_hidden_states)
            model_runner_output = self.decoder_worker.execute_model(scheduler_output, hat_batch_input)
            scheduler_output = self.hat_manager.process_outputs_enc_dec_loop(scheduler_output.scheduled_cached_reqs, model_runner_output)
    
    def _handle_empty_scheduler_output(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        self.encoder_worker.execute_model(scheduler_output)
        self.decoder_worker.execute_model(scheduler_output)
        self.backbone_worker.execute_model(scheduler_output)
        return EMPTY_MODEL_RUNNER_OUTPUT
    
    def _setup_modules(self):
        self.encoder_connector = self.encoder_worker.get_model().encoder_connector
        self.hat_manager.first_word_embedding = (
            self.backbone_worker.get_model().decoder_connector.first_word_embedding.squeeze(0)
        )

    @property
    def rank(self):
        return self.encoder_worker.rank

    @property
    def device(self):
        return self.encoder_worker.device

    @property
    def driver_rank(self) -> int:
        return 0


class HATModelWorker(Worker):

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        hat_batch_input: Optional[HATBatchInput] = None,
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors,
                                                 hat_batch_input)
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        return output if self.is_driver_worker else None
            
        
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
    assert layer_names == set(kv_cache_raw_tensors.keys(
    )), "Some layers are not correctly initialized"
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
    for i, kv_cache_group_spec in enumerate(
            kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        for layer_name in kv_cache_group_spec.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = (raw_tensor.numel() //
                            kv_cache_spec.page_size_bytes)
            if isinstance(kv_cache_spec, AttentionSpec):
                kv_cache_shape = attn_backends[i].get_kv_cache_shape(
                    num_blocks, kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                dtype = kv_cache_spec.dtype
                try:
                    kv_cache_stride_order = attn_backends[
                        i].get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(
                        kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(
                        range(len(kv_cache_shape)))
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
                kv_caches[layer_name] = kv_cache_raw_tensors[
                    layer_name].view(dtype).view(kv_cache_shape).permute(
                        *inv_order)
            else:
                raise NotImplementedError
    return kv_caches
    
        
