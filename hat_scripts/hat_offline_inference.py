import argparse
from vllm import LLM, SamplingParams
import torch
from hat_utils import allowed_hat_models, prompts_128

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=allowed_hat_models,
        help="HAT model to run.",
        default=allowed_hat_models[0],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=range(1, len(prompts_128)),
        help="Batch size to run.",
        default=16,
    )
    parser.add_argument(
        "--max-bytes-per-req",
        type=int,
        help="Maximum output bytes per request",
        default=1000,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size.",
        default=1,
    )
    return parser.parse_args()

format_llama = lambda s: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>

{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    
if __name__ == "__main__":
    args = parse_args()
    
    llm = LLM(model=args.model,
          trust_remote_code=True,
          dtype=torch.bfloat16,
          enforce_eager=False,
          compilation_config={"full_cuda_graph": True, "level": 0},
          tensor_parallel_size=args.tensor_parallel_size,
          gpu_memory_utilization=0.9,
          max_num_batched_tokens=100000,
          max_num_seqs=128)

    prompts = prompts_128[:args.batch_size]
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=args.max_bytes_per_req)

    # To profile, you must specify VLLM_TORCH_PROFILER_DIR in the environment.
    # llm.start_profile()
    outputs = llm.generate([format_llama(p) for p in prompts], sampling_params)
    # llm.stop_profile()

    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"--- Prompt {idx+1} ---")
        print(f"{prompt}\n")
        
        print(f"--- Generation for Prompt {idx+1} ---")
        print(generated_text)
        print("-" * 50)
