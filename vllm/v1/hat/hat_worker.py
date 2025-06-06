from vllm.config import VllmConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.worker.worker_base import WorkerBase
from transformers import PretrainedConfig
import copy


def create_hat_worker(*args, **kwargs):
    ModelRegistry.register_model(
        "HATDecoderForCausalLM", 
        "vllm.model_executor.models.hat:HATDecoderForCausalLM"
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
    pass