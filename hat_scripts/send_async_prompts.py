import aiohttp
import argparse
import asyncio
import json
from hat_utils import prompts_128

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        type=str,
        help="URL of the OpenAI-compatible chat completions API endpoint (e.g., http://localhost:8000/v1/chat/completions)",
        default=API_URL,
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        choices=range(1, len(prompts_128) + 1),
        help="Number of concurrent requests to send.",
        default=16,
    )
    parser.add_argument(
        "--max-bytes-per-req",
        type=int,
        help="Maximum output bytes per request.",
        default=1000,
    )
    return parser.parse_args()

async def send_single_prompt_async(session, prompt_text, prompt_index, api_url, max_bytes_per_req, total_prompts):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. You give engaging, well-structured answers to user inquiries."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_bytes_per_req,
        "temperature": 0.8
    }
    print(f"Sending prompt {prompt_index + 1}/{total_prompts}: '{prompt_text[:50]}...'")
    try:
        async with session.post(api_url, headers=HEADERS, data=json.dumps(payload)) as response:
            response.raise_for_status()
            response_data = await response.json()
            print(f"Received response for prompt {prompt_index + 1}")
            return response_data
    except aiohttp.ClientError as e:
        print(f"Error sending request for prompt {prompt_index + 1} ('{prompt_text[:50]}...'): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for prompt {prompt_index + 1} ('{prompt_text[:50]}...'): {e}")
        return None

async def main():
    args = parse_args()
    
    # Slice prompts_128 to get the requested concurrent requests
    prompts_to_use = prompts_128[:args.num_concurrent_requests]
    
    print(f"Processing {len(prompts_to_use)} prompts with max_bytes_per_req={args.max_bytes_per_req}")
    print(f"Using API URL: {args.api_url}")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts_to_use):
            tasks.append(send_single_prompt_async(session, prompt, i, args.api_url, args.max_bytes_per_req, len(prompts_to_use)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            original_prompt = prompts_to_use[i]
            print(f"--- Prompt {i+1} ---")
            print(f"{original_prompt}\n")
            
            print(f"--- Generation for Prompt {i+1} ---")
            if isinstance(result, Exception):
                error_message = f"An error occurred: {result}"
                print(error_message)
            elif result:
                response_json = json.dumps(result, indent=2)
                print(response_json)
            else:
                no_data_message = "No response data received."
                print(no_data_message)
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 