from typing import Optional
from vllm.config import VllmConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.worker.worker_base import WorkerBase
from transformers import PretrainedConfig
import copy


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
    # kwargs["model_runner_cls"] = HATModelRunner
    worker_class = resolve_obj_by_qualname("vllm.v1.worker.gpu_worker.Worker")
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
        self.decoder_model_runner = decoder_worker.model_runner
        self.backbone_model_runner = backbone_worker.model_runner
        self.encoder_connector = None
        self.use_cuda_graph = vllm_config.compilation_config.max_capture_size > 0
        self.hat_manager = None
        self.pred_word_embeds_buffer = None
        self.vllm_config = vllm_config
        self.backbone_num_gpu_blocks: Optional[int] = None
        self.process_full_word = False
        
    def init_device(self) -> None:
        self.encoder_worker.init_device()
        self.encoder_worker.load_model()
        self.decoder_worker.init_device()
        self.decoder_worker.load_model()
        self.backbone_worker.init_device()
        self.backbone_worker.load_model()
    
    def load_model(self) -> None:
        pass

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.encoder_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                             num_cpu_blocks=num_cpu_blocks)
        self.decoder_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                             num_cpu_blocks=num_cpu_blocks)
        self.backbone_worker.initialize_cache(num_gpu_blocks=self.backbone_num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)