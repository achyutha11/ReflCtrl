import argparse
import copy
import json
import os
from typing import List, Dict, Tuple
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import subprocess
import tempfile
from scipy.optimize import minimize
from transformers import AutoTokenizer
from arg_utils import add_common_arguments
from hook_utils import InterventionDirection, MODEL_NUM_LAYERS_MAP
from query_llm import MODELS
from utils import get_save_dir
from multiprocessing import Pool

def get_probe_save_dir(model: str, dataset: str, intervention_direction: str, randomize: bool = False, use_last_token_embedding: bool = False) -> str:
    """
    Get the directory path for saving/loading probe data.
    
    Args:
        model: Name of the model
        dataset: Name of the dataset
        intervention_direction: Direction of intervention
        randomize: Whether random intervention was used
        use_last_token_embedding: Whether to use last token embeddings instead of probe directions
        
    Returns:
        str: Path to the probe data directory
    """
    if use_last_token_embedding:
        probe_save_dir = f"probe_data/{model}/{dataset}/last_token_embedding"
    else:
        probe_save_dir = f"probe_data/{model}/{dataset}/{intervention_direction}"
        if randomize:
            probe_save_dir += "_random"
    return probe_save_dir

def get_intervention_dir(model: str, dataset: str, intervention_direction: str, randomize: bool = False) -> Tuple[str, InterventionDirection]:
    """
    Load intervention direction from file.
    
    Args:
        model: Name of the model
        dataset: Name of the dataset
        intervention_direction: Direction of intervention
        randomize: Whether to randomize the intervention direction
        
    Returns:
        Tuple containing:
        - str: Path to the intervention direction file
        - InterventionDirection: Loaded intervention direction object
    """
    intervention_path = f"intervention_direction/{model}/{dataset}/{intervention_direction}_dir.pt"
    intervention_dir = InterventionDirection.load(intervention_path)
    
    if randomize:
        for component in intervention_dir.components:
            orig_shape = intervention_dir.components[component].mean_diff.shape
            intervention_dir.components[component].mean_diff = torch.randn(orig_shape).type_as(intervention_dir.components[component].mean_diff)
    
    return intervention_path, intervention_dir

def extract_features_labels(probe_data: List[Tuple[torch.Tensor, bool]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and labels from probe data.
    
    Args:
        probe_data: List of tuples containing (feature, label)
        
    Returns:
        Tuple containing:
        - torch.Tensor: Stacked features
        - torch.Tensor: Labels tensor
    """
    features = torch.stack([x[0] for x in probe_data])
    labels = torch.tensor([x[1] for x in probe_data])
    return features, labels

def get_think_end_token(full_prompt: str, prompt_tokens: List[int], probe_results: Dict, think_end_token: int) -> torch.Tensor:
    think_end_token_index = prompt_tokens.index(think_end_token)
    features = []
    for key in sorted(probe_results.keys()):
        try:
            features.append(probe_results[key][0][think_end_token_index].item())
        except:
            print(f"Error at {key}")
            print(full_prompt)
            print(probe_results[key])
            print(think_end_token_index)
            print(len(prompt_tokens))
            raise Exception("Error")
    features = torch.Tensor(features)
    return features


def get_average(full_prompt: str, prompt_tokens: List[int], probe_results: Dict, think_end_token: int) -> torch.Tensor:
    think_end_token_index = prompt_tokens.index(think_end_token)
    features = []
    for key in sorted(probe_results.keys()):
        features.append(probe_results[key][0][:think_end_token_index].mean().item())
    features = torch.Tensor(features)
    return features


def get_average_think_steps(full_prompt: str, prompt_tokens: List[int], probe_results: Dict, think_end_token: int, think_delim_tokens: List[int]) -> torch.Tensor:
    think_end_token_index = prompt_tokens.index(think_end_token)
    think_steps_positions = [i for i, tid in enumerate(prompt_tokens[:think_end_token_index]) if tid in think_delim_tokens]
    features = []   
    for key in sorted(probe_results.keys()):
        features.append(probe_results[key][0][think_steps_positions].mean().item())
    features = torch.Tensor(features)
    return features


def get_last_think_step(full_prompt: str, prompt_tokens: List[int], probe_results: Dict, think_end_token: int, think_delim_tokens: List[int], K=1) -> torch.Tensor:
    think_end_token_index = prompt_tokens.index(think_end_token)
    think_steps_positions = [i for i, tid in enumerate(prompt_tokens[:think_end_token_index]) if tid in think_delim_tokens]
    features = []   
    for key in sorted(probe_results.keys()):
        features.append(probe_results[key][0][think_steps_positions[-K]].item())
    features = torch.Tensor(features)
    return features

def load_outputs(dataset: str, model: str, instruction: str, with_intervention: float = 0.0,
                intervention_direction: str = "reflect", intervention_layers: str = None,
                step_begin_only: bool = False, intervention_type: str = "additive",
                n_samples: int = 1) -> Tuple[List[str], List[Dict]]:
    """
    Load model outputs from the saved results file.
    
    Returns:
        Tuple containing:
        - List of questions
        - List of response dictionaries with content, reasoning, and is_correct
    """
    save_dir = get_save_dir(dataset, model, instruction, with_intervention, 
                           intervention_direction, intervention_layers, step_begin_only, intervention_type)
    
    results_file = f"{save_dir}/results_samples{n_samples}.json"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        analyzed_results = json.load(f)
    
    # Extract data from analyzed_results format
    questions = analyzed_results["questions"]
    sample_results = analyzed_results["sample_results"][0]  # Take first sample's results
    
    # Convert to response format
    responses = []
    for content, reasoning, is_correct in zip(sample_results["response_texts"],
                                              sample_results["think_texts"],
                                              sample_results["correctness"]):
        response = {
            "content": content,
            "reasoning": reasoning,
            "is_correct": is_correct
        }
        responses.append(response)
    
    return questions, responses


def collect_raw_probe_data(
    llm: LLM,
    questions: List[str],
    responses: List[Dict],
    instruction: str,
    probe: object,
    sampling_params: SamplingParams,
) -> List[Tuple[Dict, List[int], str, bool]]:
    """
    Collect raw probe data for each model output.
    
    Args:
        llm: The LLM instance
        questions: List of input questions
        responses: List of model responses
        instruction: Instruction to append to questions
        probe: The probe object attached to model
        sampling_params: SamplingParams instance for generation
        
    Returns:
        List of tuples containing:
        - probe_results: Dict of probe data
        - prompt_tokens: List of token IDs
        - full_prompt: Complete prompt string
        - is_correct: Boolean indicating correctness
    """
    raw_probe_data = []
    tokenizer = llm.get_tokenizer()
    
    for question, response in zip(questions, responses):
        # Create input by combining question and response
        content = response["content"]
        is_correct = response["is_correct"]
        reasoning = response["reasoning"]

        # Format the prompt using the template
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question+instruction}], tokenize=False, add_generation_prompt=True)
        
        # Clear probe cache and generate a single token to collect probe data
        probe.clear_cache()
        if reasoning is not None:
            full_prompt = prompt + reasoning + "\n</think>\n" + content
        else:
            full_prompt = prompt + content
        response = llm.generate(full_prompt, sampling_params)
        
        # Get probe results
        probe_results = probe.get_cache()
        for key in probe_results.keys():
            if len(probe_results[key]) > 1:
                probe_results[key][0] = torch.cat(probe_results[key], dim=0)
            assert probe_results[key][0].shape[0] == len(response[0].prompt_token_ids), \
                  f"Probe results shape: {[tensor.shape for tensor in probe_results[key]]}, response shape: {len(response[0].prompt_token_ids)}"
        # Store raw results
        raw_probe_data.append((copy.deepcopy(probe_results), response[0].prompt_token_ids, full_prompt, is_correct))
        
    return raw_probe_data

class LastTokenEmbeddingHook():
    """Hook to capture the last token's or last thinking token's hidden state from the final layer."""
    def __init__(self, act_store, use_last_thinking_token=False, think_end_token=None, use_prompt_embedding=False):
        self.act_store = act_store
        self.use_last_thinking_token = use_last_thinking_token
        self.think_end_token = think_end_token
        self.use_prompt_embedding = use_prompt_embedding
        self.current_prompt_tokens = None
        self.prompt_end_position = None
    
    def set_prompt_tokens(self, prompt_tokens):
        """Set the current prompt tokens to find think_end_token position."""
        self.current_prompt_tokens = prompt_tokens
    
    def set_prompt_end_position(self, prompt_end_position):
        """Set the position where the original prompt ends (before model response)."""
        self.prompt_end_position = prompt_end_position
    
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]  # Usually the first element is hidden states
        else:
            hidden_states = output
        
        # Extract token embedding: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        if len(hidden_states.shape) == 3:
            if self.use_prompt_embedding and self.prompt_end_position is not None:
                # Take the embedding at the end of the prompt (before model response)
                token_position = min(self.prompt_end_position, hidden_states.shape[1] - 1)
                token_embedding = hidden_states[:, token_position, :].cpu()
            elif self.use_last_thinking_token and self.think_end_token is not None and self.current_prompt_tokens is not None:
                # Find the position of the think_end_token
                try:
                    think_end_token_index = self.current_prompt_tokens.index(self.think_end_token)
                    # Take the token right before </think>
                    token_position = max(0, think_end_token_index)
                    token_embedding = hidden_states[:, token_position, :].cpu()
                except (ValueError, IndexError):
                    # Fallback to last token if think_end_token not found
                    token_embedding = hidden_states[:, -1, :].cpu()
            else:
                # Take last token (original behavior)
                token_embedding = hidden_states[:, -1, :].cpu()
        else:
            # Handle case where batch dimension might be squeezed
            token_embedding = hidden_states[-1, :].cpu() if len(hidden_states.shape) == 2 else hidden_states.cpu()
        
        # Convert to float32 if in BFloat16 to avoid numpy conversion issues later
        if token_embedding.dtype == torch.bfloat16:
            token_embedding = token_embedding.float()
        
        self.act_store.append(token_embedding)


class LastTokenEmbeddingCacher():
    """Cacher specifically for capturing last token embeddings from the final layer."""
    def __init__(self, use_last_thinking_token=False, think_end_token=None, use_prompt_embedding=False):
        self.cache = []
        self.handle = None
        self.hook = None
        self.use_last_thinking_token = use_last_thinking_token
        self.think_end_token = think_end_token
        self.use_prompt_embedding = use_prompt_embedding
    
    def register_model(self, model):
        """Register hook on the last layer of the model."""
        # Find the last layer - typically model.layers[-1] for transformer models
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # For models like Llama/Qwen
            last_layer = model.model.layers[-1]
            target_module_name = f"model.layers[{len(model.model.layers)-1}]"
        elif hasattr(model, 'layers'):
            last_layer = model.layers[-1]
            target_module_name = f"layers[{len(model.layers)-1}]"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # For GPT-style models
            last_layer = model.transformer.h[-1]
            target_module_name = f"transformer.h[{len(model.transformer.h)-1}]"
        else:
            raise ValueError("Could not find transformer layers in the model")
        
        # Register hook on the last layer
        self.hook = LastTokenEmbeddingHook(self.cache, self.use_last_thinking_token, self.think_end_token, self.use_prompt_embedding)
        self.handle = last_layer.register_forward_hook(self.hook)
        
        if self.use_prompt_embedding:
            hook_type = "prompt end"
        elif self.use_last_thinking_token:
            hook_type = "last thinking token"
        else:
            hook_type = "last token"
        print(f"Registered {hook_type} embedding hook on: {target_module_name}")
    
    def set_prompt_tokens(self, prompt_tokens):
        """Set the current prompt tokens for the hook."""
        if self.hook is not None:
            self.hook.set_prompt_tokens(prompt_tokens)
    
    def set_prompt_end_position(self, prompt_end_position):
        """Set the prompt end position for the hook."""
        if self.hook is not None:
            self.hook.set_prompt_end_position(prompt_end_position)
        
    def get_cache(self):
        return self.cache
    
    def clear_cache(self):
        self.cache.clear()
        
    def remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class AllLayerEmbeddingHook():
    """Hook to capture hidden states from a specific layer."""
    def __init__(self, act_store, layer_idx, use_last_thinking_token=False, think_end_token=None, use_prompt_embedding=False):
        self.act_store = act_store
        self.layer_idx = layer_idx
        self.use_last_thinking_token = use_last_thinking_token
        self.think_end_token = think_end_token
        self.use_prompt_embedding = use_prompt_embedding
        self.current_prompt_tokens = None
        self.prompt_end_position = None
    
    def set_prompt_tokens(self, prompt_tokens):
        """Set the current prompt tokens to find think_end_token position."""
        self.current_prompt_tokens = prompt_tokens
    
    def set_prompt_end_position(self, prompt_end_position):
        """Set the position where the original prompt ends (before model response)."""
        self.prompt_end_position = prompt_end_position
    
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]  # Usually the first element is hidden states
        else:
            hidden_states = output
        
        # Extract token embedding: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        if len(hidden_states.shape) == 3:
            if self.use_prompt_embedding and self.prompt_end_position is not None:
                # Take the embedding at the end of the prompt (before model response)
                token_position = min(self.prompt_end_position, hidden_states.shape[1] - 1)
                token_embedding = hidden_states[:, token_position, :].cpu()
            elif self.use_last_thinking_token and self.think_end_token is not None and self.current_prompt_tokens is not None:
                # Find the position of the think_end_token
                try:
                    think_end_token_index = self.current_prompt_tokens.index(self.think_end_token)
                    # Take the token right before </think>
                    token_position = max(0, think_end_token_index)
                    token_embedding = hidden_states[:, token_position, :].cpu()
                except (ValueError, IndexError):
                    # Fallback to last token if think_end_token not found
                    token_embedding = hidden_states[:, -1, :].cpu()
            else:
                # Take last token (original behavior)
                token_embedding = hidden_states[:, -1, :].cpu()
        else:
            # Handle case where batch dimension might be squeezed
            token_embedding = hidden_states[-1, :].cpu() if len(hidden_states.shape) == 2 else hidden_states.cpu()
        
        # Convert to float32 if in BFloat16 to avoid numpy conversion issues later
        if token_embedding.dtype == torch.bfloat16:
            token_embedding = token_embedding.float()
        
        self.act_store[self.layer_idx] = token_embedding


class AllLayerEmbeddingCacher():
    """Cacher specifically for capturing hidden states from all layers."""
    def __init__(self, use_last_thinking_token=False, think_end_token=None, use_prompt_embedding=False):
        self.cache = {}
        self.handles = []
        self.hooks = []
        self.use_last_thinking_token = use_last_thinking_token
        self.think_end_token = think_end_token
        self.use_prompt_embedding = use_prompt_embedding
    
    def register_model(self, model):
        """Register hooks on all layers of the model."""
        # Find all layers - typically model.layers for transformer models
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # For models like Llama/Qwen
            layers = model.model.layers
            layer_prefix = "model.layers"
        elif hasattr(model, 'layers'):
            layers = model.layers
            layer_prefix = "layers"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # For GPT-style models
            layers = model.transformer.h
            layer_prefix = "transformer.h"
        else:
            raise ValueError("Could not find transformer layers in the model")
        
        # Register hook on each layer
        for layer_idx, layer in enumerate(layers):
            hook = AllLayerEmbeddingHook(self.cache, layer_idx, self.use_last_thinking_token, self.think_end_token, self.use_prompt_embedding)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)
        
        if self.use_prompt_embedding:
            hook_type = "prompt end"
        elif self.use_last_thinking_token:
            hook_type = "last thinking token"
        else:
            hook_type = "last token"
        print(f"Registered {hook_type} embedding hooks on {len(layers)} layers: {layer_prefix}[0-{len(layers)-1}]")
    
    def set_prompt_tokens(self, prompt_tokens):
        """Set the current prompt tokens for all hooks."""
        for hook in self.hooks:
            hook.set_prompt_tokens(prompt_tokens)
    
    def set_prompt_end_position(self, prompt_end_position):
        """Set the prompt end position for all hooks."""
        for hook in self.hooks:
            hook.set_prompt_end_position(prompt_end_position)
    
    def get_cache(self):
        return self.cache
    
    def clear_cache(self):
        self.cache.clear()
        
    def remove_hook(self):
        for handle in self.handles:
            if handle is not None:
                handle.remove()
        self.handles.clear()
        self.hooks.clear()


def collect_last_token_embeddings(
    llm: LLM,
    questions: List[str],
    responses: List[Dict],
    instruction: str,
    sampling_params: SamplingParams,
    use_last_thinking_token: bool = False,
    use_prompt_embedding: bool = False,
) -> List[Tuple[torch.Tensor, bool]]:
    """
    Collect last token embeddings from the final layer for each model output.
    
    Args:
        llm: The LLM instance
        questions: List of input questions
        responses: List of model responses
        instruction: Instruction to append to questions
        sampling_params: SamplingParams instance for generation
        use_last_thinking_token: Whether to use last thinking token instead of last token
        use_prompt_embedding: Whether to use prompt end embedding instead of last token
        
    Returns:
        List of tuples containing:
        - last_token_embedding: The final layer hidden state of the specified token
        - is_correct: Boolean indicating correctness
    """
    embedding_data = []
    tokenizer = llm.get_tokenizer()
    
    # Get think_end_token if using last thinking token
    think_end_token = None
    if use_last_thinking_token:
        think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    
    # Create and register the last token embedding cacher using apply_model
    cacher = LastTokenEmbeddingCacher(use_last_thinking_token, think_end_token, use_prompt_embedding)
    llm.apply_model(lambda model: cacher.register_model(model))
    
    for question, response in zip(questions, responses):
        # Create input by combining question and response
        content = response["content"]
        reasoning = response["reasoning"]
        is_correct = response["is_correct"]

        # Format the prompt using the template
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question+instruction}], tokenize=False, add_generation_prompt=True)
        
        # Clear cache before each generation
        cacher.clear_cache()
        
        full_prompt = prompt + reasoning + "\n</think>\n" + content
        
        # Set prompt tokens for the hook if using last thinking token
        if use_last_thinking_token:
            # Tokenize the full prompt to get token positions
            prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
            cacher.set_prompt_tokens(prompt_tokens)
        
        # Set prompt end position if using prompt embedding
        if use_prompt_embedding:
            # Calculate the position where the original prompt ends
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_end_position = len(prompt_tokens) - 1  # 0-indexed position
            cacher.set_prompt_end_position(prompt_end_position)
        
        # Generate to trigger the hooks and capture last token embedding
        outputs = llm.generate(full_prompt, sampling_params)
        
        # Get the captured last token embedding
        cache = cacher.get_cache()
        if len(cache) == 0:
            raise RuntimeError("No embeddings were captured. Check if hooks are properly registered.")
        
        # The cache contains the last token embedding from the final layer
        last_token_embedding = cache[0]  # Should be [hidden_dim] tensor
        
        # Handle batch dimension if present
        if len(last_token_embedding.shape) > 1:
            last_token_embedding = last_token_embedding[0]  # Take first in batch
        
        # Store the embedding and correctness
        embedding_data.append((last_token_embedding, is_correct))
    
    # Clean up hooks
    cacher.remove_hook()
    
    return embedding_data


def collect_all_layers_embeddings(
    llm: LLM,
    questions: List[str],
    responses: List[Dict],
    instruction: str,
    sampling_params: SamplingParams,
    use_last_thinking_token: bool = False,
    use_prompt_embedding: bool = False,
) -> Dict[int, List[Tuple[torch.Tensor, bool]]]:
    """
    Collect embeddings from all layers for each model output.
    
    Args:
        llm: The LLM instance
        questions: List of input questions
        responses: List of model responses
        instruction: Instruction to append to questions
        sampling_params: SamplingParams instance for generation
        use_last_thinking_token: Whether to use last thinking token instead of last token
        use_prompt_embedding: Whether to use prompt end embedding instead of last token
        
    Returns:
        Dict mapping layer index to list of tuples containing:
        - embedding: The layer's hidden state of the specified token
        - is_correct: Boolean indicating correctness
    """
    all_layers_data = {}
    tokenizer = llm.get_tokenizer()
    
    # Get think_end_token if using last thinking token
    think_end_token = None
    if use_last_thinking_token:
        think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    
    # Create and register the all layers embedding cacher using apply_model
    cacher = AllLayerEmbeddingCacher(use_last_thinking_token, think_end_token, use_prompt_embedding)
    llm.apply_model(lambda model: cacher.register_model(model))
    
    for question, response in zip(questions, responses):
        # Create input by combining question and response
        content = response["content"]
        reasoning = response["reasoning"]
        is_correct = response["is_correct"]

        # Format the prompt using the template
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question+instruction}], tokenize=False, add_generation_prompt=True)
        
        # Clear cache before each generation
        cacher.clear_cache()
        
        full_prompt = prompt + reasoning + "\n</think>\n" + content
        
        # Set prompt tokens for the hook if using last thinking token
        if use_last_thinking_token:
            # Tokenize the full prompt to get token positions
            prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
            cacher.set_prompt_tokens(prompt_tokens)
        
        # Set prompt end position if using prompt embedding
        if use_prompt_embedding:
            # Calculate the position where the original prompt ends
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_end_position = len(prompt_tokens) - 1  # 0-indexed position
            cacher.set_prompt_end_position(prompt_end_position)
        
        # Generate to trigger the hooks and capture embeddings from all layers
        outputs = llm.generate(full_prompt, sampling_params)
        
        # Get the captured embeddings from all layers
        cache = cacher.get_cache()
        if len(cache) == 0:
            raise RuntimeError("No embeddings were captured. Check if hooks are properly registered.")
        
        # Organize embeddings by layer
        for layer_idx, embedding in cache.items():
            # Handle batch dimension if present
            if len(embedding.shape) > 1:
                embedding = embedding[0]  # Take first in batch
            
            # Initialize layer data if not exists
            if layer_idx not in all_layers_data:
                all_layers_data[layer_idx] = []
            
            # Store the embedding and correctness for this layer
            all_layers_data[layer_idx].append((embedding, is_correct))
    
    # Clean up hooks
    cacher.remove_hook()
    
    return all_layers_data


def process_data_point(data_point, think_end_token, think_delim_tokens, aggregation_strategy):
    probe_results, prompt_tokens, full_prompt, is_correct = data_point
    # Aggregate features based on strategy
    if aggregation_strategy == "think_end_token":
        feature = get_think_end_token(full_prompt, prompt_tokens, probe_results, think_end_token)
    elif aggregation_strategy == "average":
        feature = get_average(full_prompt, prompt_tokens, probe_results, think_end_token)
    elif aggregation_strategy == "average_think_steps":
        feature = get_average_think_steps(full_prompt, prompt_tokens, probe_results, think_end_token, think_delim_tokens)
    elif aggregation_strategy == "last_think_step":
        feature = get_last_think_step(full_prompt, prompt_tokens, probe_results, think_end_token, think_delim_tokens, K=1)
    elif aggregation_strategy == "sec_last_think_steps":
        feature = get_last_think_step(full_prompt, prompt_tokens, probe_results, think_end_token, think_delim_tokens, K=2)
    return feature, is_correct

def aggregate_probe_data(
    raw_probe_data: List[Tuple[Dict, List[int], str, bool]],
    tokenizer,
    aggregation_strategy: str
) -> List[Tuple[torch.Tensor, bool]]:
    """
    Aggregate raw probe data using specified strategy.
    
    Args:
        raw_probe_data: List of tuples containing raw probe data
        tokenizer: Tokenizer instance
        aggregation_strategy: Strategy to use for aggregation
        
    Returns:
        List of tuples containing:
        - feature: Aggregated feature tensor
        - is_correct: Boolean indicating correctness
    """
    think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    think_delim_tokens = [tid for tid in range(tokenizer.vocab_size) if "\n\n" in tokenizer.decode(tid)]
    probe_stores = []
    for data_point in raw_probe_data:
        feature, is_correct = process_data_point(data_point, think_end_token, think_delim_tokens, aggregation_strategy)
        probe_stores.append((feature, is_correct))
    return probe_stores

def create_weight_visualization(clf: LogisticRegression, feature_names: List[str]):
    """
    Create and launch a Streamlit visualization for model weights.
    """
    weights = clf.coef_[0]  # Get weights from the model
    abs_weights = np.abs(weights)
    
    # Create a DataFrame with weights and their absolute values
    df = pd.DataFrame({
        'Component': feature_names,
        'Weight': weights,
        'Absolute Weight': abs_weights
    })
    
    # Sort by absolute weight for better visualization
    df = df.sort_values('Absolute Weight', ascending=True)
    
    # Create a temporary directory without using context manager
    temp_dir = tempfile.mkdtemp()
    
    # Save the weights data
    weights_path = os.path.join(temp_dir, 'weights.json')
    df.to_json(weights_path, orient='records')
    
    # Create the Streamlit app
    app_path = os.path.join(temp_dir, 'app.py')
    with open(app_path, 'w') as f:
        f.write("""import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import shutil
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the weights data
with open(os.path.join(current_dir, 'weights.json'), 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

st.title('Linear Probe Weight Analysis')

# Add description
st.write('This visualization shows the weights of the linear probe classifier. Larger absolute weights indicate more important components for classification.')

# Create horizontal bar chart of weights
fig = px.bar(df, 
             x='Weight', 
             y='Component',
             orientation='h',
             title='Component Weights',
             color='Weight',
             color_continuous_scale='RdBu',
             labels={'Weight': 'Weight Value', 'Component': 'Component Name'})

# Update layout for better visualization
fig.update_layout(
    height=max(400, len(df) * 20),  # Dynamic height based on number of components
    xaxis_title='Weight Value',
    yaxis_title='Component',
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Show top influential components
st.subheader('Most Influential Components')
st.write('Components with largest absolute weights:')

top_n = st.slider('Number of top components to show', min_value=5, max_value=len(df), value=10)

top_components = df.sort_values('Absolute Weight', ascending=False).head(top_n)
st.table(top_components[['Component', 'Weight', 'Absolute Weight']].round(4))

# Cleanup function
def cleanup():
    try:
        shutil.rmtree(current_dir)
    except:
        pass

# Register cleanup on session end
st.session_state.setdefault('cleanup_done', False)
if not st.session_state.cleanup_done:
    st.session_state.cleanup_done = True
    import atexit
    atexit.register(cleanup)
""")
    
    # Launch the Streamlit app
    print("\nLaunching visualization website...")
    print(f"Temporary files are stored in: {temp_dir}")
    subprocess.run(['streamlit', 'run', app_path])

def train_constrained_logistic_regression(X, y, enforce_negative=False, balance_classes=False):
    """
    Train logistic regression with optional constraint that all weights must be negative.
    """
    n_features = X.shape[1]
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def objective(params):
        if balance_classes:
            positive_freq = np.mean(y)
        w = params[:-1]  # weights
        b = params[-1]   # bias
        z = X @ w + b
        pred = sigmoid(z)
        # Log loss with L2 regularization
        if balance_classes:
            loss = -np.mean((1 - positive_freq) * (y * np.log(pred + 1e-15) + positive_freq * (1 - y) * np.log(1 - pred + 1e-15)))
        else:
            loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
        # Add L2 regularization
        loss += 0.01 * np.sum(w ** 2)  # 0.01 is the regularization strength
        return loss
    
    def grad(params):
        w = params[:-1]
        b = params[-1]
        z = X @ w + b
        pred = sigmoid(z)
        # Gradient of log loss
        dw = X.T @ (pred - y) / len(y)
        db = np.mean(pred - y)
        # Add gradient of L2 regularization
        dw += 0.02 * w  # 0.02 is 2 * regularization strength
        return np.concatenate([dw, [db]])
    
    # Initial weights and bias
    initial_params = np.zeros(n_features + 1)
    
    if enforce_negative:
        # Constraint: all weights must be negative (except bias)
        constraints = []
        for i in range(n_features):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: -x[idx],  # Constraint: w[i] <= 0
                'jac': lambda x, idx=i: np.array([-1 if j == idx else 0 for j in range(len(x))])
            })
    else:
        constraints = []
    
    # Optimize
    result = minimize(
        objective,
        initial_params,
        method='SLSQP',
        jac=grad,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    # Create a custom classifier object with similar interface to sklearn's
    class CustomLogisticRegression:
        def __init__(self, weights, bias):
            self.coef_ = weights.reshape(1, -1)
            self.intercept_ = np.array([bias])
        
        def predict(self, X):
            return (self.decision_function(X) > 0).astype(int)
        
        def decision_function(self, X):
            return X @ self.coef_[0] + self.intercept_[0]
    
    # Extract weights and bias
    weights = result.x[:-1]
    bias = result.x[-1]
    
    return CustomLogisticRegression(weights, bias)

def create_layer_performance_visualization(layer_results: Dict, save_dir: str):
    """
    Create and launch a Streamlit visualization for layer-wise performance.
    """
    # Create a DataFrame with layer performance
    data = []
    for layer_idx, results in layer_results.items():
        data.append({
            'Layer': layer_idx,
            'Train Accuracy': results['train_accuracy'],
            'Val Accuracy': results['val_accuracy'],
            'Val F1': results['val_f1'],
            'AUROC': results['au_roc'],
            'AUPR': results['au_pr']
        })
    
    df = pd.DataFrame(data).sort_values('Layer')
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save the performance data
    performance_path = os.path.join(temp_dir, 'layer_performance.json')
    df.to_json(performance_path, orient='records')
    
    # Create the Streamlit app
    app_path = os.path.join(temp_dir, 'layer_performance_app.py')
    with open(app_path, 'w') as f:
        f.write("""import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import shutil

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the performance data
with open(os.path.join(current_dir, 'layer_performance.json'), 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

st.title('Layer-wise Probe Performance Analysis')

# Add description
st.write('This visualization shows the performance of linear probes trained on embeddings from different layers of the model.')

# Create line plots for different metrics
metrics = ['Train Accuracy', 'Val Accuracy', 'Val F1', 'AUROC', 'AUPR']

# Allow user to select which metrics to display
selected_metrics = st.multiselect(
    'Select metrics to display:',
    metrics,
    default=['Val Accuracy', 'Val F1', 'AUROC']
)

if selected_metrics:
    # Create line plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=df['Layer'],
            y=df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Layer-wise Performance Comparison',
        xaxis_title='Layer Index',
        yaxis_title='Performance Score',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Show detailed performance table
st.subheader('Detailed Performance by Layer')
st.dataframe(df.round(4), use_container_width=True)

# Find and highlight best performing layers
st.subheader('Best Performing Layers')
best_layers = {}
for metric in metrics:
    best_idx = df[metric].idxmax()
    best_layer = df.loc[best_idx, 'Layer']
    best_score = df.loc[best_idx, metric]
    best_layers[metric] = (best_layer, best_score)

for metric, (layer, score) in best_layers.items():
    st.write(f"**{metric}**: Layer {layer} ({score:.4f})")

# Performance distribution
st.subheader('Performance Distribution')
metric_for_dist = st.selectbox('Select metric for distribution analysis:', metrics, index=2)

fig_hist = px.histogram(df, x=metric_for_dist, nbins=20, 
                       title=f'Distribution of {metric_for_dist} across layers')
st.plotly_chart(fig_hist, use_container_width=True)

# Cleanup function
def cleanup():
    try:
        shutil.rmtree(current_dir)
    except:
        pass

# Register cleanup on session end
st.session_state.setdefault('cleanup_done', False)
if not st.session_state.cleanup_done:
    st.session_state.cleanup_done = True
    import atexit
    atexit.register(cleanup)
""")
    
    # Launch the Streamlit app
    print("\nLaunching layer performance visualization...")
    print(f"Temporary files are stored in: {temp_dir}")
    subprocess.run(['streamlit', 'run', app_path])


def eval_all_layers_probe(args):
    """
    Evaluate probe features from all layers by training a linear classifier for each layer.
    """
    # Load saved all layers embedding data
    probe_save_dir = get_probe_save_dir(args.model, args.dataset, args.intervention_direction, args.randomize, True)  # Always use embedding mode
    
    # Determine save filename based on embedding type
    if args.use_prompt_embedding:
        save_filename = "all_layers_prompt_end_embeddings.pt"
    elif args.use_last_thinking_token:
        save_filename = "all_layers_last_thinking_token_embeddings.pt"
    else:
        save_filename = "all_layers_last_token_embeddings.pt"
    
    all_layers_data = torch.load(f"{probe_save_dir}/{save_filename}")
    
    # Get model info to determine number of layers
    model_key = args.model
    if model_key not in MODEL_NUM_LAYERS_MAP:
        print(f"Warning: Model {model_key} not found in MODEL_NUM_LAYERS_MAP. Using detected layers from data.")
        num_layers = max(all_layers_data.keys()) + 1
    else:
        num_layers = MODEL_NUM_LAYERS_MAP[model_key]
    
    print(f"Evaluating probes for {num_layers} layers...")
    
    layer_results = {}
    
    for layer_idx in range(num_layers):
        if layer_idx not in all_layers_data:
            print(f"Warning: No data found for layer {layer_idx}")
            continue
            
        print(f"\nEvaluating layer {layer_idx}...")
        
        # Extract features and labels for this layer
        layer_data = all_layers_data[layer_idx]
        features, labels = extract_features_labels(layer_data)
        
        # Convert to float32 if needed (BFloat16 is not supported by numpy)
        if features.dtype == torch.bfloat16:
            features = features.float()
        if labels.dtype == torch.bfloat16:
            labels = labels.float()
        
        # Split into train/val sets
        X_train, X_val, y_train, y_val = train_test_split(
            features.numpy(), labels.numpy(), 
            test_size=0.2, random_state=42
        )
        
        # Train logistic regression
        if args.enforce_negative:
            clf = train_constrained_logistic_regression(X_train, y_train, enforce_negative=True, balance_classes=args.balance_classes)
        else:
            clf = LogisticRegression(random_state=42, penalty="l2", class_weight="balanced" if args.balance_classes else None,
                                     max_iter=2000)
            clf.fit(X_train, y_train)
        
        # Evaluate on in-distribution validation set
        train_preds = clf.predict(X_train)
        val_preds = clf.predict(X_val)
        val_scores = clf.decision_function(X_val)
        
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='binary'
        )
        au_roc = roc_auc_score(y_val, val_scores)
        au_pr = average_precision_score(y_val, val_scores)
        
        # Save results for this layer
        layer_results[layer_idx] = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1': float(val_f1),
            'au_roc': float(au_roc),
            'au_pr': float(au_pr)
        }
        
        # Save layer-specific model weights and bias
        layer_save_dir = f"{probe_save_dir}/layer_{layer_idx}"
        os.makedirs(layer_save_dir, exist_ok=True)
        torch.save(clf.coef_, f"{layer_save_dir}/clf_weights.pt")
        torch.save(clf.intercept_, f"{layer_save_dir}/clf_bias.pt")
        
        print(f"Layer {layer_idx} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, AUROC: {au_roc:.3f}, AUPR: {au_pr:.3f}")
    
    # Save all results
    with open(f"{probe_save_dir}/all_layers_results.json", 'w') as f:
        json.dump(layer_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("LAYER-WISE PROBE PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Layer':<6} {'Train Acc':<10} {'Val Acc':<10} {'Val F1':<10} {'AUROC':<10} {'AUPR':<10}")
    print("-" * 80)
    
    for layer_idx in sorted(layer_results.keys()):
        results = layer_results[layer_idx]
        print(f"{layer_idx:<6} {results['train_accuracy']:<10.3f} {results['val_accuracy']:<10.3f} "
              f"{results['val_f1']:<10.3f} {results['au_roc']:<10.3f} {results['au_pr']:<10.3f}")
    
    # Find best performing layer
    best_layer = max(layer_results.keys(), key=lambda x: layer_results[x]['val_f1'])
    best_results = layer_results[best_layer]
    print(f"\nBest performing layer: {best_layer} (Val F1: {best_results['val_f1']:.3f})")
    
    # Create visualization if requested
    if args.visualize:
        create_layer_performance_visualization(layer_results, probe_save_dir)
    
    return layer_results


def eval_probe(args):
    """
    Evaluate probe features by training a linear classifier.
    """
    # Load saved probe data
    probe_save_dir = get_probe_save_dir(args.model, args.dataset, args.intervention_direction, args.randomize, args.use_last_token_embedding)
    
    if args.use_last_token_embedding:
        # Load last token embeddings directly based on embedding type
        if args.use_prompt_embedding:
            probe_data = torch.load(f"{probe_save_dir}/prompt_end_embeddings.pt")
        elif args.use_last_thinking_token:
            probe_data = torch.load(f"{probe_save_dir}/last_thinking_token_embeddings.pt")
        else:
            probe_data = torch.load(f"{probe_save_dir}/last_token_embeddings.pt")
    else:
        probe_data = torch.load(f"{probe_save_dir}/raw_probe.pt")
        # Aggregate probe data
        tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
        probe_data = aggregate_probe_data(
            raw_probe_data=probe_data,
            tokenizer=tokenizer,
            aggregation_strategy=args.aggregation_strategy
        )
    
    # Extract features and labels
    features, labels = extract_features_labels(probe_data)
    
    # Convert to float32 if needed (BFloat16 is not supported by numpy)
    if features.dtype == torch.bfloat16:
        features = features.float()
    if labels.dtype == torch.bfloat16:
        labels = labels.float()
    
    # Split into train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        features.numpy(), labels.numpy(), 
        test_size=0.2, random_state=42
    )
    
    # Train logistic regression
    if args.enforce_negative:
        print("Training logistic regression with negative weight constraint...")
        clf = train_constrained_logistic_regression(X_train, y_train, enforce_negative=True, balance_classes=args.balance_classes)
    else:
        clf = LogisticRegression(random_state=42, penalty="l2", class_weight="balanced" if args.balance_classes else None,
                                 max_iter=2000)
        clf.fit(X_train, y_train)
    # Save the model weight and bias
    torch.save(clf.coef_, f"{probe_save_dir}/clf_weights.pt")
    torch.save(clf.intercept_, f"{probe_save_dir}/clf_bias.pt")
    # Evaluate on in-distribution validation set
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    val_scores = clf.decision_function(X_val)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_preds, average='binary'
    )
    au_roc = roc_auc_score(y_val, val_scores)
    au_pr = average_precision_score(y_val, val_scores)
    
    # Save results
    results = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'au_roc': float(au_roc),
        'au_pr': float(au_pr)
    }
    
    # Evaluate on OOD dataset if specified
    if args.ood_dataset:
        print(f"\nEvaluating on OOD dataset: {args.ood_dataset}")
        
        # Load OOD probe data
        ood_save_dir = get_probe_save_dir(args.model, args.ood_dataset, args.intervention_direction, args.randomize, args.use_last_token_embedding)
        
        try:
            if args.use_last_token_embedding:
                # Load last token embeddings directly based on embedding type
                if args.use_prompt_embedding:
                    ood_probe_data = torch.load(f"{ood_save_dir}/prompt_end_embeddings.pt")
                elif args.use_last_thinking_token:
                    ood_probe_data = torch.load(f"{ood_save_dir}/last_thinking_token_embeddings.pt")
                else:
                    ood_probe_data = torch.load(f"{ood_save_dir}/last_token_embeddings.pt")
            else:
                ood_probe_data = torch.load(f"{ood_save_dir}/raw_probe.pt")
                # Aggregate probe data
                ood_probe_data = aggregate_probe_data(
                    raw_probe_data=ood_probe_data,
                    tokenizer=tokenizer,
                    aggregation_strategy=args.aggregation_strategy
                )
            # Extract features and labels
            ood_features, ood_labels = extract_features_labels(ood_probe_data)
            
            # Convert to float32 if needed (BFloat16 is not supported by numpy)
            if ood_features.dtype == torch.bfloat16:
                ood_features = ood_features.float()
            if ood_labels.dtype == torch.bfloat16:
                ood_labels = ood_labels.float()
                
            ood_features = ood_features.numpy()
            ood_labels = ood_labels.numpy()
            
            # Evaluate
            ood_preds = clf.predict(ood_features)
            ood_scores = clf.decision_function(ood_features)
            
            ood_acc = accuracy_score(ood_labels, ood_preds)
            ood_precision, ood_recall, ood_f1, _ = precision_recall_fscore_support(
                ood_labels, ood_preds, average='binary'
            )
            ood_au_roc = roc_auc_score(ood_labels, ood_scores)
            ood_au_pr = average_precision_score(ood_labels, ood_scores)
            
            # Add OOD results
            results.update({
                'ood_accuracy': float(ood_acc),
                'ood_precision': float(ood_precision),
                'ood_recall': float(ood_recall),
                'ood_f1': float(ood_f1),
                'ood_au_roc': float(ood_au_roc),
                'ood_au_pr': float(ood_au_pr)
            })
            
            print(f"OOD Accuracy: {ood_acc:.3f}")
            print(f"OOD F1-Score: {ood_f1:.3f}")
            print(f"OOD AUROC: {ood_au_roc:.3f}")
            print(f"OOD AUPR: {ood_au_pr:.3f}")
            # Save ood scores
            torch.save(ood_scores, f"{probe_save_dir}/{args.ood_dataset}_scores.pt")

        except FileNotFoundError:
            print(f"Warning: Could not find probe data for OOD dataset at {ood_save_dir}")
            print("Make sure to run extract mode on the OOD dataset first.")
    
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    print(f"Validation F1-Score: {val_f1:.3f}")
    print(f"Validation AUROC: {au_roc:.3f}")
    print(f"Validation AUPR: {au_pr:.3f}")
    
    # Create visualization if requested
    if args.visualize:
        if args.use_last_token_embedding:
            # Generate feature names for embedding dimensions
            feature_names = [f"embed_dim_{i}" for i in range(len(clf.coef_[0]))]
        else:
            # Generate feature names based on feature dimension
            intervention_path, intervention_dir = get_intervention_dir(args.model, args.itv_dataset, args.intervention_direction)
            feature_names = list(sorted(intervention_dir.components.keys()))
        create_weight_visualization(clf, feature_names)

def main(args):
    # Validate arguments
    if args.use_last_thinking_token and not args.use_last_token_embedding:
        raise ValueError("--use_last_thinking_token can only be used with --use_last_token_embedding")
    
    if args.use_prompt_embedding and not args.use_last_token_embedding:
        raise ValueError("--use_prompt_embedding can only be used with --use_last_token_embedding")
    
    if args.use_prompt_embedding and args.use_last_thinking_token:
        raise ValueError("--use_prompt_embedding and --use_last_thinking_token cannot be used together")
    
    if args.use_last_token_embedding and args.mode in ["extract", "extract_all_layers"]:
        if args.use_prompt_embedding:
            embedding_type = "prompt end"
        elif args.use_last_thinking_token:
            embedding_type = "last thinking token"
        else:
            embedding_type = "last token"
        print(f"Using {embedding_type} embeddings as features instead of probe directions")
    
    # Additional validation for all-layers modes
    if args.mode in ["extract_all_layers", "eval_all_layers"] and not args.use_last_token_embedding:
        raise ValueError("All layers modes require --use_last_token_embedding to be enabled")
        
    if args.mode in ["extract", "extract_all_layers"]:
        # Load saved outputs
        questions, responses = load_outputs(
            args.dataset, args.model, args.instruction, args.with_intervention,
            args.intervention_direction, args.intervention_layers, args.step_begin_only, args.intervention_type, args.n_samples
        )
        
        # Initialize model
        llm = LLM(
            model=MODELS[args.model],
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_length,
            enforce_eager=True
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=1,  # Only need to generate one token to get probe data
            top_p=0.95
        )
        
        if args.mode == "extract_all_layers":
            # Collect embeddings from all layers
            all_layers_data = collect_all_layers_embeddings(
                llm=llm,
                questions=questions,
                responses=responses,
                instruction=args.instruction,
                sampling_params=sampling_params,
                use_last_thinking_token=args.use_last_thinking_token,
                use_prompt_embedding=args.use_prompt_embedding
            )
            # Save all layers embedding data
            probe_save_dir = get_probe_save_dir(args.model, args.dataset, args.intervention_direction, args.randomize, True)
            os.makedirs(probe_save_dir, exist_ok=True)
            
            # Determine save filename based on embedding type
            if args.use_prompt_embedding:
                save_filename = "all_layers_prompt_end_embeddings.pt"
            elif args.use_last_thinking_token:
                save_filename = "all_layers_last_thinking_token_embeddings.pt"
            else:
                save_filename = "all_layers_last_token_embeddings.pt"
            
            torch.save(all_layers_data, f"{probe_save_dir}/{save_filename}")
            print(f"Saved embeddings from {len(all_layers_data)} layers to {probe_save_dir}/{save_filename}")
            
        elif args.use_last_token_embedding:
            # Collect last token embeddings instead of probe data
            embedding_data = collect_last_token_embeddings(
                llm=llm,
                questions=questions,
                responses=responses,
                instruction=args.instruction,
                sampling_params=sampling_params,
                use_last_thinking_token=args.use_last_thinking_token,
                use_prompt_embedding=args.use_prompt_embedding
            )
            # Save embedding data
            probe_save_dir = get_probe_save_dir(args.model, args.dataset, args.intervention_direction, args.randomize,
                                                args.use_last_token_embedding)
            os.makedirs(probe_save_dir, exist_ok=True)
            
            # Determine save filename based on embedding type
            if args.use_prompt_embedding:
                save_filename = "prompt_end_embeddings.pt"
            elif args.use_last_thinking_token:
                save_filename = "last_thinking_token_embeddings.pt"
            else:
                save_filename = "last_token_embeddings.pt"
            
            torch.save(embedding_data, f"{probe_save_dir}/{save_filename}")
        else:
            # Load intervention direction
            intervention_path, intervention_dir = get_intervention_dir(args.model, args.itv_dataset, args.intervention_direction, args.randomize)
            args.intervention_path = intervention_path
            
            probe = llm.apply_model(lambda model: intervention_dir.add_prober(model))[0]
            
            # Collect raw probe data
            raw_probe_data = collect_raw_probe_data(
                llm=llm,
                questions=questions,
                responses=responses,
                instruction=args.instruction,
                probe=probe,
                sampling_params=sampling_params
            )
            # Save raw probe data
            probe_save_dir = get_probe_save_dir(args.model, args.dataset, args.intervention_direction, args.randomize, args.use_last_token_embedding)
            os.makedirs(probe_save_dir, exist_ok=True)
            torch.save(raw_probe_data, f"{probe_save_dir}/raw_probe.pt")
    
    elif args.mode == "eval":
        eval_probe(args)
    
    elif args.mode == "eval_all_layers":
        eval_all_layers_probe(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and evaluate probe data from model outputs")
    parser.add_argument("--itv_dataset", type=str, default="gsm8k",
                      help="Intervention dataset to use")
    parser.add_argument("--mode", type=str, choices=["extract", "eval", "extract_all_layers", "eval_all_layers"], default="extract",
                      help="Mode to run: extract features, evaluate probe, extract all layers, or evaluate all layers")
    parser.add_argument("--ood_dataset", type=str, default="",
                      help="Out-of-distribution dataset to evaluate on (eval mode only)")
    parser.add_argument("--aggregation_strategy", type=str, 
                        default="think_end_token",
                        choices=["think_end_token", "average", "average_think_steps", "sec_last_think_steps", "last_think_step"],
                        help="Aggregation strategy to use")
    parser.add_argument("--visualize", action="store_true",
                      help="Create a web app to visualize model weights (eval mode only)")
    parser.add_argument("--enforce_negative", action="store_true",
                      help="Enforce all weights to be negative in the linear probe")
    parser.add_argument("--randomize", action="store_true",
                      help="Replace intervention direction with random tensors of same size")
    parser.add_argument("--balance_classes", action="store_true",
                      help="Balance the classes in the training set")
    parser.add_argument("--n_samples", type=int, default=1,
                      help="Number of samples to collect")
    parser.add_argument("--use_last_token_embedding", action="store_true",
                      help="Use last token embedding as features instead of probe directions")
    parser.add_argument("--use_last_thinking_token", action="store_true",
                      help="Use last thinking token embedding instead of last token (only when --use_last_token_embedding is enabled)")
    parser.add_argument("--use_prompt_embedding", action="store_true",
                      help="Use prompt end embedding instead of last token (only when --use_last_token_embedding is enabled)")
    # Add common arguments with intervention options
    add_common_arguments(parser, include_intervention=True)
    
    args = parser.parse_args()
    main(args) 