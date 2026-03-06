import argparse
from collections import defaultdict
import json
import os
import pickle
import warnings
import torch
import hashlib
from hook_utils import MODEL_ATTN_LAYER_MAP, ActivationCacher, MODEL_LAYER_MAP, Qwen2AttentionActivationCacher
from vllm import LLM, SamplingParams, TokensPrompt
from utils import END_WORDS, extract_questions, MODELS, REFLECT_WORDS
from typing import List, Dict
from arg_utils import add_common_arguments




def collect_activations(questions: List[str], model: str, instruction: str, tensor_parallel_size: int = 1, get_headwise_activations: bool = False) -> List[Dict]:
    """
    Run offline batch inference using vLLM directly.
    
    Args:
        questions: List of questions to process
        model: Name of the model to use
        n_samples: Number of samples to generate per question
        
    Returns:
        List of response dictionaries
    """
    # Initialize the LLM
    llm = LLM(model=MODELS[model], tensor_parallel_size=tensor_parallel_size,
            max_model_len=MAX_RESPONSE_LENGTH+2048,
            enforce_eager=True)
    if get_headwise_activations:
        activation_cacher = Qwen2AttentionActivationCacher()
    else:
        activation_cacher = ActivationCacher()
    tokenizer = llm.get_tokenizer()
    THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
    THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]
    THINK_DELIM_TOKEN_IDS = [tid for tid in range(tokenizer.vocab_size) if "\n\n" in tokenizer.decode(tid)]
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.6,
                                    max_tokens=MAX_RESPONSE_LENGTH,
                                    top_p=0.95,
                                    stop_token_ids=[THINK_END_TOKEN_ID])
    output_list = []
    # Precompute the outputs without hooks
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": question + instruction}], tokenize=False, add_generation_prompt=True) for question in questions]
    outputs_full_ids = []
    # Cache outputs according to model, questions and instruction
    cache_key = f"{model}_{hashlib.md5(str(questions).encode()).hexdigest()}_{hashlib.md5(instruction.encode()).hexdigest()}"
    cache_file = f"cache/{cache_key}.pkl"
    os.makedirs("cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        print(f"Loading cached outputs from {cache_file}")
        with open(cache_file, "rb") as f:
            outputs_full_ids = pickle.load(f)
    else:
        print(f"Caching outputs to {cache_file}")
        outputs = llm.generate(prompts, sampling_params)
        outputs_full_ids = []
        for prompt, output in zip(prompts, outputs):
            prompt_token_ids = output.prompt_token_ids
            output_token_ids = output.outputs[0].token_ids
            text = output.outputs[0].text
            outputs_full_ids.append((prompt_token_ids, output_token_ids, text))
            output_list.append({"prompt": prompt, "output": text})
        with open(cache_file, "wb") as f:
            pickle.dump(outputs_full_ids, f)
    # Rerun the inference with hooks
    target_modules = MODEL_LAYER_MAP[model] if not get_headwise_activations else MODEL_ATTN_LAYER_MAP[model]
    llm.apply_model(lambda x: activation_cacher.register_model(x, target_modules))
    
    if get_headwise_activations:
        # For headwise activations, use running mean to save memory
        activation_means = {}
        activation_counts = {}
    else:
        # For regular activations, use list accumulation as before
        activation_stores = defaultdict(list)
    
    is_reflect_stores = []
    is_end_stores = []
    sampling_params.max_tokens = 1
    for i, (prompt_token_ids, output_token_ids, text) in enumerate(outputs_full_ids):
        # Generate responses for all questions at once
        activation_cacher.clear_cache()
        token_prompt = TokensPrompt(prompt_token_ids=list(prompt_token_ids) + list(output_token_ids))
        outputs = llm.generate(token_prompt, sampling_params)
        offset = len(prompt_token_ids)
        activations = activation_cacher.get_cache()
        # Split the text according to \n\n
        think_steps = text.split("\n\n")
        # Identify the position of \n\n tokens
        think_step_positions = [-1] + [i for i, token_id in enumerate(output_token_ids) if token_id in THINK_DELIM_TOKEN_IDS]
        if len(think_step_positions) != len(think_steps):
            warnings.warn(f"Number of think steps ({len(think_steps)}) does not match number of think step positions ({len(think_step_positions)}) for question {questions[i+j]}")
            print(text)
            print(output_token_ids)
            continue
        # Collect activations for all think steps first
        current_activations = {}
        for think_step, think_step_position in zip(think_steps, think_step_positions):
            is_reflect = 1 if any(word in think_step.lower() for word in REFLECT_WORDS) else 0
            is_reflect_stores.append(is_reflect)
            is_end = 1 if any(word in think_step.lower() for word in END_WORDS) else 0
            is_end_stores.append(is_end)
            for key in activations.keys():
                assert activations[key][0].shape[0] == len(output_token_ids) + len(prompt_token_ids), f"Activation shape mismatch: {activations[key][0].shape[0]} != {len(output_token_ids) + len(prompt_token_ids)}"
                current_activation = activations[key][0][offset + think_step_position, ...].clone()
                
                if key not in current_activations:
                    current_activations[key] = []
                current_activations[key].append(current_activation)
        
        # Now update the stores after all steps are collected
        for key, activations_list in current_activations.items():
            if get_headwise_activations:
                # Stack all activations for this key and update running mean in batch
                stacked_activations = torch.stack(activations_list, dim=0)  # Shape: [num_steps, ...]
                
                if key not in activation_means:
                    activation_means[key] = stacked_activations.mean(dim=0)
                    activation_counts[key] = len(activations_list)
                else:
                    # Batch running mean update
                    old_count = activation_counts[key]
                    new_count = old_count + len(activations_list)
                    
                    # Weight the old mean and new batch mean by their respective counts
                    activation_means[key] = (activation_means[key] * old_count + stacked_activations.sum(dim=0)) / new_count
                    activation_counts[key] = new_count
            else:
                # For regular activations, append as before
                for current_activation in activations_list:
                    activation_stores[key].append(current_activation[None, ...])
    if get_headwise_activations:
        # Convert running means to final activation stores
        activation_stores = activation_means
    else:
        # Concatenate regular activations as before
        for key in activation_stores.keys():
            activation_stores[key] = torch.cat(activation_stores[key], dim=0)
    return activation_stores, is_reflect_stores, is_end_stores, output_list    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect activations from model inference")
    
    # Add common arguments with debug flag
    add_common_arguments(parser)
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with limited samples")
    parser.add_argument("--get_headwise_activations", action="store_true",
                       help="Get headwise activations")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Limit to first N questions (default: all)")
    args = parser.parse_args()
    MAX_RESPONSE_LENGTH = args.max_length
    questions = extract_questions(args.dataset)
    if args.debug:
        questions = questions[:10]
    elif args.n_samples is not None:
        questions = questions[:args.n_samples]
    activation_stores, is_reflect_stores, is_end_stores, output_list = collect_activations(questions, args.model, args.instruction, args.tensor_parallel_size, args.get_headwise_activations)
    output_dir = f"activations/{args.model}/{args.dataset}/{args.instruction}"
    if args.get_headwise_activations:
        output_dir += "/headwise"
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"activation_stores": activation_stores, "is_reflect_stores": is_reflect_stores, "is_end_stores": is_end_stores}, f"{output_dir}/activations.pt")
    with open(f"{output_dir}/outputs.jsonl", "w") as f:
        for output in output_list:
            f.write(json.dumps(output) + "\n")