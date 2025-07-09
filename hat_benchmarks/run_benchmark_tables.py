from vllm import LLM, SamplingParams
import torch
import time
import numpy as np
import sys
import os
from hat_splitter import HATSplitter
import argparse
from enum import Enum
import json
import os
from typing import Dict, List
import glob

format_llama = lambda s: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>
{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

format_llama_long = lambda s: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a versatile and expert AI assistant, capable of being a master storyteller, a brilliant educator, and a profound thinker. Your purpose is to provide exceptionally detailed, insightful, and engaging responses to a wide array of creative and explanatory prompts.
For creative and descriptive tasks: Immerse the user in the world you are building. Use vivid, multi-sensory language (sights, sounds, smells, textures) to make scenes and concepts come alive. Develop compelling characters, intricate plots, and rich, consistent lore. Your descriptions should be cinematic and evocative.
For explanatory and technical tasks: Become the world's clearest communicator. Break down complex subjects into understandable, digestible concepts. Use insightful analogies and structured, step-by-step explanations to ensure clarity. Prioritize accuracy and detail, providing specific data like temperatures or stages when requested.
For philosophical or design tasks: Delve deep. Present nuanced arguments, consider multiple viewpoints, and explore the subtle implications of the topic. When designing systems, strategies, or worlds, focus on internal logic, creativity, and comprehensive detail.
In all responses, strive for depth, coherence, and a touch of brilliance. Your goal is not just to answer the prompt, but to inspire, educate, and captivate the user with the quality and richness of your response.Please remember to always be aligned with human values and be sure to delight the user.<|eot_id|><|start_header_id|>user<|end_header_id|>
{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

max_bytes = 1000
batch_sizes_to_consider = [1, 2, 4, 8, 16, 32, 64]
max_prompts_in_files = max(batch_sizes_to_consider)

data = glob.glob("language_prompts*.json")
dict_language_to_prompts: Dict[str, List[str]] = {}

for file in data:
    with open(file, "r") as f:
        dict_language_to_prompts.update(json.load(f))
                
datasets_no_formatting_needed = ["pga", "pga_filtered"]

class ModelName(Enum):
    HAT = "hat"
    LLAMA = "llama"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=ModelName,
        choices=list(ModelName),
        help="name of model [hat, llama]",
        default=ModelName.HAT,
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=list(dict_language_to_prompts.keys()),
        help="language of prompts",
        default="english",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        choices=batch_sizes_to_consider,
        help="max batch size",
        default=max(batch_sizes_to_consider),
    )
    parser.add_argument(
        "--average",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="average results over N runs",
    )
    parser.add_argument(
        "--long-system-prompt",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Uses a longer system prompt",
    )
    return parser.parse_args()


def get_max_tokens(model: ModelName, language: str, max_bytes: int) -> int:
    match model:
        case ModelName.HAT:
            return max_bytes
        case ModelName.LLAMA:
            if language == "english":
                return int(max_bytes / 4.5)
            elif language == "german" or language == "pga" or language == "pga_filtered":
                return int(max_bytes / 3.5)
            else:
                return int(max_bytes / 5)


def get_model_path(model_name: ModelName) -> str:
    match model_name:
        case ModelName.HAT:
            return "/nfs/scratch-aa/hat_vllm/dpo"
        case ModelName.LLAMA:
            return "/nfs/checkpoint-tuning/llama3_hf/Meta-Llama-3.1-8B-Instruct/"    
        
        
def average_over_n_runs(batch_size: int) -> int:
    if batch_size >= 128:
        return 1
    else:
        return 10
    
    
if __name__ == "__main__":
    args = parse_args()
    model = args.model
    language = args.language
    max_batch_size = args.max_batch_size
    average = args.average
    long_system_prompt = args.long_system_prompt
    
    max_tokens = get_max_tokens(model, language, max_bytes)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=max_tokens)

    hat_splitter = HATSplitter()
            
    # Initialize LLM
    model_path = get_model_path(model)
    llm = LLM(model=model_path,
              trust_remote_code=True,
              dtype=torch.bfloat16,
              enforce_eager=False, 
              compilation_config={"full_cuda_graph": True, "level": 0},
              tensor_parallel_size=1,
              gpu_memory_utilization=0.9,
              block_size=16,
              disable_cascade_attn=True,
              max_model_len=130000,
              max_num_batched_tokens=130000,
              max_num_seqs=128) # Can be set to 100k on A100
    
    tokenizer = llm.get_tokenizer()
    
    prompts = dict_language_to_prompts[language]
    assert len(prompts) >= max_prompts_in_files
    if language not in datasets_no_formatting_needed:
        if long_system_prompt:
            prompts = [format_llama_long(p) for p in prompts]
            print(f"Using long system prompt for {language}")
        else:
            prompts = [format_llama(p) for p in prompts]
            print(f"Using normal system prompt for {language}")
    else:
        print(f"No formatting needed for {language}")

    # sampling_params is assumed to be defined globally.
    # max_tokens is also assumed to be defined globally and used in sampling_params.
    print(f"Starting benchmark with max_tokens_per_request = {max_tokens}")
    print("="*70)

    results_summary = []
    batch_sizes = [batch_size for batch_size in batch_sizes_to_consider if batch_size <= max_batch_size]
    for actual_batch_size in batch_sizes:
        name = f"batch_size_{actual_batch_size}"
        runs_to_average_over = 1 if not average else average_over_n_runs(actual_batch_size)
        
        print("\n\n\n\n")
        print("-"*70)
        print("-"*70)
        print(f"Averaging batch size {actual_batch_size} over {runs_to_average_over} runs")
        print("-"*70)
        print("-"*70)
        list_mean_input_bytes = []
        list_mean_input_tokens = []
        list_time_spent_s = []
        list_mean_output_bytes = []
        list_mean_output_tokens = []
        list_bytes_per_second = []
        list_bytes_per_second_per_request = []
        list_compression_bytes_per_token = []
        list_len_bytes_input_per_seq = []
        list_len_bytes_generation_per_seq = []

        for i in range(runs_to_average_over):
            random_idxs = np.random.choice(len(prompts), size=actual_batch_size, replace=False)
            prompts_to_use = [prompts[idx] for idx in random_idxs]
            
            len_bytes_input_per_seq: List[int] = []
            
            print("\n\n\n\n")
            print(f"Batch size: {actual_batch_size} | Run {i}")

            len_bytes_input_per_seq = [len(p.encode('utf-8')) for p in prompts_to_use]
            total_input_bytes = sum(len_bytes_input_per_seq)
            mean_input_bytes = total_input_bytes / actual_batch_size if actual_batch_size > 0 else 0
                
            match model:
                case ModelName.HAT:
                    total_input_tokens = sum(len(tokenizer.encode_to_words(p)) for p in prompts_to_use)
                case ModelName.LLAMA:
                    total_input_tokens = sum(len(tokenizer.encode(p)) for p in prompts_to_use) # P add_special_tokens=False
            mean_input_tokens = total_input_tokens / actual_batch_size if actual_batch_size > 0 else 0                
            
            start_time = time.time()
            try:
                outputs = llm.generate(prompts_to_use, sampling_params)
            except Exception as e:
                print(f"  Error during llm.generate for {name}: {e}. Skipping.")
                continue
            end_time = time.time()
            print("Representative output:")
            print(outputs[0].outputs[0].text)

            time_spent = end_time - start_time
            total_generated_bytes_for_batch = 0
            total_generated_tokens_for_batch = 0
            len_bytes_generation_per_seq: List[int] = []
            for output_seq in outputs:
                if output_seq.outputs:
                    completion_output = output_seq.outputs[0]
                    output_bytes = len(completion_output.text.encode('utf-8'))
                    len_bytes_generation_per_seq.append(output_bytes)
                    total_generated_bytes_for_batch += output_bytes
                    match model:
                        case ModelName.HAT:
                            total_generated_tokens_for_batch += len(tokenizer.encode_to_words(completion_output.text))
                        case ModelName.LLAMA:
                            total_generated_tokens_for_batch += len(tokenizer.encode(completion_output.text))
                else:
                    len_bytes_generation_per_seq.append(0)
                    print(f"  Warning: No output found for a sequence in batch {name}.")
            
            mean_output_bytes = total_generated_bytes_for_batch / actual_batch_size if actual_batch_size > 0 else 0
            mean_output_tokens = total_generated_tokens_for_batch / actual_batch_size if actual_batch_size > 0 else 0
            
            bytes_per_second = 0
            if time_spent > 0:
                bytes_per_second = total_generated_bytes_for_batch / time_spent # P also include input bytes
            
            bytes_per_second_per_request = 0
            if actual_batch_size > 0 and isinstance(bytes_per_second, (int, float)) and bytes_per_second > 0:
                bytes_per_second_per_request = bytes_per_second / actual_batch_size
            else:
                bytes_per_second_per_request = "N/A"
                
            compression_bytes_per_token = mean_output_bytes / mean_output_tokens
            
            list_mean_input_bytes.append(mean_input_bytes)
            list_mean_input_tokens.append(mean_input_tokens)
            list_time_spent_s.append(time_spent)
            list_mean_output_bytes.append(mean_output_bytes)
            list_mean_output_tokens.append(mean_output_tokens)
            list_bytes_per_second.append(bytes_per_second)
            list_bytes_per_second_per_request.append(bytes_per_second_per_request)
            list_compression_bytes_per_token.append(compression_bytes_per_token)
            list_len_bytes_input_per_seq.append(len_bytes_input_per_seq)
            list_len_bytes_generation_per_seq.append(len_bytes_generation_per_seq)
        print("-"*70)
        
        # We now calculate stdev or std error of list_bytes_per_second and list_bytes_per_second_per_request 
        bytes_per_second_std = np.std(list_bytes_per_second)
        bytes_per_second_se = bytes_per_second_std / np.sqrt(runs_to_average_over) if runs_to_average_over > 0 else 0.0
        bytes_per_second_per_request_std = np.std(list_bytes_per_second_per_request)
        bytes_per_second_per_request_se = bytes_per_second_per_request_std / np.sqrt(runs_to_average_over) if runs_to_average_over > 0 else 0.0
        
        results_summary.append({
            "name": name,
            "batch_size": actual_batch_size,
            "mean_input_bytes": np.mean(list_mean_input_bytes),
            "mean_input_tokens": np.mean(list_mean_input_tokens),
            "time_spent_s": np.mean(list_time_spent_s),
            "mean_output_bytes": np.mean(list_mean_output_bytes),
            "mean_output_tokens": np.mean(list_mean_output_tokens),
            "bytes_per_second": np.mean(list_bytes_per_second),
            "bytes_per_second_std_error": bytes_per_second_se,
            "bytes_per_second_per_request": np.mean(list_bytes_per_second_per_request),
            "bytes_per_second_per_request_std_error": bytes_per_second_per_request_se,
            "compression_bytes_per_token": np.mean(list_compression_bytes_per_token),
            "status": "Success",
            "len_bytes_input_per_seq": list_len_bytes_input_per_seq,
            "len_bytes_generation_per_seq": list_len_bytes_generation_per_seq,
            "runs_averaged_over": runs_to_average_over,
        })
        
    print("\nBenchmark Summary:")
    print("="*185) # Adjusted width for new table column
    header = f"{'Test Name':<15} | {'Batch Size':<10} | {'Mean In Bytes':<15} | {'Time (s)':<10} | {'Mean Out Bytes':<15} | {'Out Bytes/s':<12} | {'Out B/s/Req':<12} | {'Mean In Tokens':<15} | {'Mean Out Tokens':<15} | {'Compression [B/T]':<15} | {'Status':<15}" # Added new column header
    print(header)
    print("-"*(len(header)))
    for res in results_summary:
        name_val = res.get('name','N/A')
        bs_val = res.get('batch_size','N/A')
        in_bytes_val = res.get('mean_input_bytes','N/A')
        time_val = res.get('time_spent_s','N/A')
        out_bytes_val = res.get('mean_output_bytes','N/A')
        out_bps_val = res.get('bytes_per_second','N/A')
        out_bps_req_val = res.get('bytes_per_second_per_request', 'N/A') # Get new metric
        status_val = res.get('status','N/A')

        time_str = f"{time_val:.3f}s" if isinstance(time_val, (int, float)) else str(time_val)
        in_bytes_str = f"{in_bytes_val:.2f}" if isinstance(in_bytes_val, (int, float)) else str(in_bytes_val)
        out_bytes_str = f"{out_bytes_val:.2f}" if isinstance(out_bytes_val, (int, float)) else str(out_bytes_val)
        out_bps_str = f"{out_bps_val:.2f}" if isinstance(out_bps_val, (int, float)) else str(out_bps_val)
        out_bps_req_str = f"{out_bps_req_val:.2f}" if isinstance(out_bps_req_val, (int, float)) else str(out_bps_req_val) # Format new metric
        in_tokens_val = res.get('mean_input_tokens','N/A')
        out_tokens_val = res.get('mean_output_tokens','N/A')
        compression_bytes_per_token_val = res.get('compression_bytes_per_token','N/A')
        compression_str = f"{compression_bytes_per_token_val:.2f}" if isinstance(compression_bytes_per_token_val, (int, float)) else str(compression_bytes_per_token_val)

        print(f"{str(name_val):<15} | {str(bs_val):<10} | {in_bytes_str:<15} | {time_str:<10} | {out_bytes_str:<15} | {out_bps_str:<12} | {out_bps_req_str:<12}  | {in_tokens_val:<15} | {out_tokens_val:<15} | {compression_str:<15} | {str(status_val)[:15]:<15}")# Print new metric
    print("="*185) # Adjusted width

    # Save results to JSON file
    # Create nested dictionary structure
    json_results = {language: {}}
    
    for res in results_summary:
        batch_size = res.get('batch_size', 'N/A')
        batch_key = f"batch_size_{batch_size}"
        
        # Extract all metrics except 'name' and 'batch_size' since those are used for structure
        metrics = {k: v for k, v in res.items() if k not in ['name', 'batch_size']}
        json_results[language][batch_key] = metrics
    
    # Save to JSON file
    system_prompt_used = "normal" if not args.long_system_prompt else "long"
    json_filename = f"benchmark_results_{model.value}_{language}_maxbs{max_batch_size}_system_prompt_{system_prompt_used}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {json_filename}")