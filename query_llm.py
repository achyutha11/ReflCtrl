import argparse
import asyncio
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import aiohttp
import aiohttp.client_exceptions

from utils import analyze_math_results, extract_questions, get_save_dir
# Add constants for retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1  # Base delay in seconds
MAX_DELAY = 10  # Maximum delay in seconds
# Add new constants for rate limiting
REQUEST_DELAY = 0.1  # Delay between requests in seconds
MAX_CONCURRENT_REQUESTS = 50


async def query_llm_api(question: str, session: aiohttp.ClientSession, model: str, instruction: str, n_samples: int = 1,
                        with_intervention: float = 0.0, intervention_layers: str = None, max_response_length: int = 1024,
                        no_think: bool = False, api_url: str = "http://localhost:8088/v1/chat/completions",
                        component_type: str = None, intervention_type: str = None) -> Dict:
    """
    Query the LLM API.
    """
    # Use provided API URL instead of hardcoded default
    url = api_url
    
    headers = {"Content-Type": "application/json"}
    if instruction:
        question = f"Question: {question} {instruction}"
    if with_intervention:
        model = model + f"_intv={with_intervention}"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.6,
        "top_p": 0.95,
        "n": n_samples,
        "max_completion_tokens": max_response_length,
        "no_think": no_think,
    }
    
    # Add intervention parameters if specified
    if intervention_layers is not None:
        data["intervention_layers"] = intervention_layers
    if component_type is not None:
        data["component_type"] = component_type
    if intervention_type is not None:
        data["intervention_type"] = intervention_type
    
    async with session.post(url, headers=headers, json=data) as response:
        response.raise_for_status()
        result = await response.json()
    return result


async def get_server_args(api_url: str) -> Dict:
    """Fetch server command line arguments from the API."""
    # Extract base URL from the API endpoint
    args_url = api_url.replace("chat/completions", "args")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(args_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("launch_arguments", {})
                else:
                    print(f"Warning: Failed to fetch server arguments. Status code: {response.status}")
                    return {}
    except aiohttp.client_exceptions.ClientError as e:
        print(f"Warning: Failed to fetch server arguments: {e}")
        return {}


def process_responses(responses: List[Dict]) -> List[Dict]:
    """
    Extract relevant information from LLM responses.
    
    Args:
        responses: List of raw responses from the LLM
        
    Returns:
        List of processed responses with extracted information
    """
    processed = []
    for resp in responses:
        if resp is None:
            processed.append({
                "success": False,
                "error": "Failed to get response"
            })
            continue
            
        try:
            message = resp["choices"][0]["message"]
            processed.append({
                "success": True,
                "reasoning": message.get("reasoning_content", ""),
                "content": message.get("content", ""),
                "thinking_length": message.get("thinking_length", 0)
            })
        except (KeyError, IndexError) as e:
            processed.append({
                "success": False,
                "error": f"Failed to parse response: {e}"
            })
            
    return processed

async def process_api_requests(questions: List[str], model: str, instruction: str, n_samples: int = 1,
                               with_intervention=0, intervention_layers: str = None, max_response_length: int = 1024,
                               no_think: bool = False, api_url: str = "http://localhost:8088/v1/chat/completions",
                               component_type: str = None, intervention_type: str = None) -> List[Dict]:
    """
    Process API requests asynchronously with load balancing.
    """    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=18000)) as session:
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def limited_query(question: str, index: int):
            async with semaphore:  # This limits concurrent requests
                await asyncio.sleep(REQUEST_DELAY)  # Add delay between requests
                return await query_llm_api(question, session, model, instruction, n_samples=n_samples,
                                           with_intervention=with_intervention, intervention_layers=intervention_layers,
                                           max_response_length=max_response_length, no_think=no_think, api_url=api_url,
                                           component_type=component_type, intervention_type=intervention_type)
        
        # Create tasks for all questions
        tasks = [
            limited_query(question, i)
            for i, question in enumerate(questions)
        ]
        
        # Process all tasks together while maintaining order
        responses = [None] * len(questions)
        failed_indices = []
        
        # Use gather to maintain order of responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results in order
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                failed_indices.append(i)
                responses[i] = None
            else:
                # Convert API response format to match our expected format
                samples = []
                for choice in result["choices"]:
                    samples.append({
                        "choices": [{
                            "message": choice["message"]
                        }]
                    })
                responses[i] = samples
        
        # Retry failed requests sequentially
        if failed_indices:
            print(f"\nRetrying {len(failed_indices)} failed requests sequentially...")
            for idx in failed_indices:
                question = questions[idx]
                for attempt in range(MAX_RETRIES):
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 2), MAX_DELAY * 2)
                    try:
                        response = await query_llm_api(question, session, model, instruction, n_samples=n_samples, intervention_layers=intervention_layers, max_response_length=max_response_length, no_think=no_think, api_url=api_url, component_type=component_type, intervention_type=intervention_type)
                        if response is not None:
                            # Convert API response format
                            samples = []
                            for choice in response["choices"]:
                                samples.append({
                                    "choices": [{
                                        "message": choice["message"]
                                    }]
                                })
                            responses[idx] = samples
                            print(f"Successfully retried request for question index {idx}")
                            break
                        await asyncio.sleep(delay)
                    except Exception as e:
                        print(f"Retry attempt {attempt + 1} failed for question index {idx}: {e}")
                        if attempt == MAX_RETRIES - 1:
                            print(f"All retries failed for question index {idx}")
        
        return responses

async def async_main(dataset: str, model: str, instruction: str, n_samples: int,
                     with_intervention: float = 0.1, intervention_type: str = "additive",
                     intervention_direction: str = "reflect", intervention_layers: str = None, step_begin_only: bool = False,
                     save_dir_suffix: str = "", max_response_length: int = 1024, nowait: bool = False, no_think: bool = False,
                     api_url: str = "http://localhost:8088/v1/chat/completions", intv_path: str = None,
                     component_type: str = None):
    # Get questions from dataset
    questions = extract_questions(dataset)
    
    responses = await process_api_requests(questions, model, instruction, n_samples,
                                           with_intervention=with_intervention,
                                           intervention_layers=intervention_layers,
                                           max_response_length=max_response_length,
                                           no_think=no_think,
                                           api_url=api_url,
                                           component_type=component_type,
                                           intervention_type=intervention_type)

    # Process responses for each sample
    processed_responses = [process_responses([resp[i] for resp in responses if resp is not None]) 
                         for i in range(n_samples)]
    
    stats, analyzed_results = analyze_math_results(processed_responses, dataset)
    print(stats)
    analyzed_results["instruction"] = instruction
    analyzed_results["questions"] = questions
    
    # Get base save directory
    save_dir = get_save_dir(dataset, model, instruction, with_intervention, intervention_direction,
                            intervention_layers, step_begin_only, intervention_type, nowait, intv_path)
    if no_think:
        save_dir = save_dir + "/no_think"
    if component_type:
        save_dir = save_dir + f"/{component_type}"
    if save_dir_suffix:
        save_dir = save_dir + f"/{save_dir_suffix}"
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save analysis results
    json.dump(analyzed_results, open(f"{save_dir}/results_samples{n_samples}.json", "w"))
    
    # Save command line arguments
    args_dict = {
        "client_args": {
            "dataset": dataset,
            "model": model,
            "instruction": instruction,
            "n_samples": n_samples,
            "with_intervention": with_intervention,
            "intervention_type": intervention_type,
            "intervention_direction": intervention_direction,
            "intervention_layers": intervention_layers,
            "step_begin_only": step_begin_only,
            "max_response_length": max_response_length,
            "nowait": nowait,
            "no_think": no_think,
            "api_url": api_url,
            "intv_path": intv_path,
            "component_type": component_type
        }
    }
    
    server_args = await get_server_args(api_url)
    args_dict["server_args"] = server_args
    
    json.dump(args_dict, open(f"{save_dir}/arguments.json", "w"), indent=2)

def main(dataset: str, model: str, instruction: str, n_samples: int,
         with_intervention: float = 0.1, intervention_type: str = "additive",
         intervention_direction: str = "reflect", intervention_layers: str = None, step_begin_only: bool = False,
         save_dir_suffix: str = "", max_response_length: int = 1024, nowait: bool = False, no_think: bool = False,
         api_url: str = "http://localhost:8088/v1/chat/completions", intv_path: str = None,
         component_type: str = None):
    """
    Entry point that runs the async main function.
    """
    asyncio.run(async_main(dataset, model, instruction, n_samples,
                           with_intervention, intervention_type,
                           intervention_direction, intervention_layers, step_begin_only,
                           save_dir_suffix, max_response_length=max_response_length, nowait=nowait,
                           no_think=no_think, api_url=api_url, intv_path=intv_path,
                           component_type=component_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query LLM using the API")
    parser.add_argument("--max_concurrent_requests", type=int, default=50,
                          help="Maximum number of concurrent requests (default: 50)")
    parser.add_argument("--save_dir_suffix", type=str, default="",
                          help="Suffix for the save directory")
    parser.add_argument("--no-think", dest="no_think", action="store_true", help="Disable thinking by adding an empty thinking block in requests and save under a separate directory")
    parser.add_argument("--api_url", type=str, default="http://localhost:8088/v1/chat/completions",
                        help="Full URL to POST for API mode (default: http://localhost:8088/v1/chat/completions)")
    parser.add_argument("--component_type", type=str, choices=["mlp", "attention"], default=None,
                        help="Type of component to apply intervention to (mlp or attention). If not specified, applies to both.")
    from arg_utils import add_common_arguments
    
    # Add common arguments with all optional groups enabled
    add_common_arguments(parser, 
                        include_samples=True,
                        include_intervention=True)
    
    args = parser.parse_args()
    
    # Set global SERVER_PORTS from command line argument
    MAX_CONCURRENT_REQUESTS = args.max_concurrent_requests
    main(args.dataset, args.model, args.instruction,
         args.n_samples, args.with_intervention,
         args.intervention_type, args.intervention_direction, intervention_layers=args.intervention_layers,
         step_begin_only=args.step_begin_only, save_dir_suffix=args.save_dir_suffix,
         max_response_length=args.max_length, nowait=args.nowait, no_think=args.no_think, api_url=args.api_url,
         intv_path=args.intv_path, component_type=args.component_type)
