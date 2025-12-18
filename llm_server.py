import argparse
import asyncio
import json
import os
import time
import logging
import csv
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any

# Import uvicorn and FastAPI frameworks
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import vLLM components
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
from vllm.utils import random_uuid
from vllm.logits_process import NoBadWordsLogitsProcessor
from contextlib import asynccontextmanager

# Import custom utilities
from hook_utils import InterventionDirection, HeadInterventionManager
from utils import MODELS, get_think_length

NOWAIT_TARGET_LIST = ["wait", "alternatively", "hmm", "but", "however", "alternative", "another", "check", "double-check", "oh", "maybe", "verify", "other", "again", "now", "ah", "any"]
MAX_FINISH_TOKENS = 1024
# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app):
    # Initialize app state variables for lazy model loading
    app.state.engine = None
    app.state.tokenizer = None
    app.state.think_start_token_id = None
    app.state.think_end_token_id = None
    app.state.intervention_dir = None
    app.state.head_manager = None
    app.state.initialized_model = None
    app.state.intv_weight = None
    app.state.current_intervention_layers = None
    app.state.nowait_ids = None  # Store pre-computed nowait token IDs
    
    # args should be set by the main function before uvicorn starts
    if hasattr(app, '_state') and hasattr(app._state, 'args'):
        app.state.args = app._state.args
        print(f"Command line arguments loaded successfully")
    else:
        print(f"WARNING: Command line arguments not found in app state")
        app.state.args = None
    
    print("Server started, waiting for first request to initialize model")
    yield

app = FastAPI(title="LLM Server with OpenAI API Compatibility", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables are now initialized in the lifespan function

# Request models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.6
    top_p: float = 0.95
    n: int = 1
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None
    intervention_type: Optional[str] = None
    no_think: bool = Field(default=False, alias="no-think")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    n: int = 1
    best_of: Optional[int] = None
    stream: bool = False
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None
    no_think: bool = Field(default=False, alias="no-think")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

class TokenizeRequest(BaseModel):
    model: str
    prompt: str
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Response models
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=random_uuid)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=random_uuid)
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage

class TokenizeResponse(BaseModel):
    tokens: List[int]
    token_strings: List[str]

def parse_model_name(model_string: str):
    """
    Parse the model string format "<model_name>-intv=<intervention_weight>" or "<model_name>"
    Returns a tuple of (model_name, intervention_weight)
    """
    model_name = model_string
    intervention_weight = 0.0
    
    if "_" in model_string:
        parts = model_string.split("_")
        if len(parts) > 1 and parts[-1].startswith("intv="):
            model_name = "_".join(parts[:-1])
            try:
                intervention_weight = float(parts[-1].split("=")[1])
            except (ValueError, IndexError):
                pass
    
    return model_name, intervention_weight

def parse_disabled_heads_csv(csv_file_path: str) -> List[tuple]:
    """
    Parse a CSV file containing disabled heads information.
    Expected CSV format: layer_idx,head_idx (one head per row)
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        List of tuples in format [(layer_idx, [head_idx_list])]
    """
    if not csv_file_path or not os.path.exists(csv_file_path):
        return []
    
    disabled_heads_dict = {}
    
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if required columns exist
            if 'layer_idx' not in reader.fieldnames or 'head_idx' not in reader.fieldnames:
                print(f"Error: CSV file must contain 'layer_idx' and 'head_idx' columns")
                return []
            
            for row in reader:
                layer_idx = int(row['layer_idx'])
                head_idx = int(row['head_idx'])
                
                if layer_idx not in disabled_heads_dict:
                    disabled_heads_dict[layer_idx] = []
                disabled_heads_dict[layer_idx].append(head_idx)
        
        # Convert to list of tuples format expected by HeadInterventionManager
        disabled_heads_list = [(layer_idx, head_list) for layer_idx, head_list in disabled_heads_dict.items()]
        
        print(f"Loaded disabled heads from {csv_file_path}: {disabled_heads_list}")
        return disabled_heads_list
        
    except Exception as e:
        print(f"Error parsing disabled heads CSV file {csv_file_path}: {str(e)}")
        return []

async def initialize_async_llm(model_name: str, tensor_parallel_size: int = 1, 
                  max_model_len: int = 4096, with_intervention: float = 0.0,
                  intervention_type: str = "additive", intervention_direction: str = "reflect",
                  intervention_layers: Optional[str] = None, step_begin_only: bool = False,
                  disabled_heads_csv: Optional[str] = None, head_modify_mode: str = "disable",
                  intv_path: Optional[str] = None, component_type: Optional[str] = None,
                  normalize_steer_vec: bool = False):
    """Initialize the AsyncLLMEngine with optional intervention and head disabling."""
    # Create AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=MODELS[model_name],
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    
    # Initialize the AsyncLLMEngine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Get tokenizer first - must await the coroutine
    tokenizer = await engine.get_tokenizer()
    
    # Now handle the special tokens
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    think_start_token_id = think_start_tokens[0] if think_start_tokens else None
    think_end_token_id = think_end_tokens[0] if think_end_tokens else None
    try:
        if intv_path is not None:
            # Load intervention direction from custom path
            intervention_dir = InterventionDirection.load(intv_path)
        else:
            # Load intervention direction from default path
            intervention_dir = InterventionDirection.load(f"intervention_direction/{model_name}/gsm8k/{intervention_direction}_dir.pt")
    except Exception as e:
        print(f"Error loading intervention direction: {str(e)}")
        intervention_dir = None
    
    # Parse disabled heads from CSV if provided
    disabled_heads_list = parse_disabled_heads_csv(disabled_heads_csv) if disabled_heads_csv else []
    head_manager = None
    
    # Apply intervention if requested
    if with_intervention != 0:
        if intervention_layers is not None:
            intervention_start, intervention_end = intervention_layers.split("-")
            # Select components based on component_type
            if component_type == "mlp":
                intervention_components = ["model.layers[{i}].mlp".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
            elif component_type == "attention":
                intervention_components = ["model.layers[{i}].self_attn".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
            else:  # None or invalid value - use both as default
                intervention_components = ["model.layers[{i}].mlp".format(i=i) for i in range(int(intervention_start), int(intervention_end))] \
                    + ["model.layers[{i}].self_attn".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
        else:
            intervention_components = None
            
        if step_begin_only:
            intervention_tokens = [tid for tid in range(tokenizer.vocab_size) if "\n\n" in tokenizer.decode(tid)]
        else:
            intervention_tokens = None
            
        # Apply model intervention to the AsyncLLMEngine's model
        def intervention_fn(model):
            print(model)
            return intervention_dir.add_intervention(model, with_intervention,
                                                components=intervention_components,
                                                type=intervention_type,
                                                condition_tokens=intervention_tokens,
                                                step_token_ids=intervention_tokens,
                                                normalize_steer_vec=normalize_steer_vec)
        engine.engine.model_executor.apply_model(intervention_fn)
        print(f"Intervention added with strength {with_intervention} and layers {intervention_layers}")
    
    # Apply head disabling if requested
    if disabled_heads_list:
        if head_modify_mode == "disable":
            head_manager = HeadInterventionManager(disabled_heads_list, direction=intervention_dir)
        elif head_modify_mode == "modify":
            head_manager = HeadInterventionManager(disabled_heads_list, direction=intervention_dir, mode="modify")
        
        def head_disable_fn(model):
            return head_manager.add_intervention(model)
        
        engine.engine.model_executor.apply_model(head_disable_fn)
        print(f"Head disabling applied: {len(disabled_heads_list)} layer(s) affected")
    
    print(f"Special tokens - THINK_START_TOKEN_ID: {think_start_token_id}, THINK_END_TOKEN_ID: {think_end_token_id}")
    
    return engine, tokenizer, think_start_token_id, think_end_token_id, intervention_dir, head_manager

async def lazy_initialize_model(app, model_string: str, intervention_layers: Optional[str] = None, component_type: Optional[str] = None, intervention_type: Optional[str] = None):
    """
    Initialize the model if it hasn't been initialized yet.
    Reject requests for different models if already initialized.
    Returns a tuple of (is_success, error_message) 
    """
    # Check if args are available
    if not hasattr(app.state, 'args') or app.state.args is None:
        return False, "Server not properly initialized. Missing command line arguments."
    
    # If already initialized, check if the requested model matches
    if app.state.initialized_model is not None:
        # Parse the requested model string
        requested_model_name, requested_intv_weight = parse_model_name(model_string)
        initialized_model_name, _ = parse_model_name(app.state.initialized_model)
        
        # Check if the model_name matches (ignore intervention weight for now)
        if requested_model_name != initialized_model_name:
            return False, f"Server already initialized with model '{app.state.initialized_model}'. Cannot switch to '{model_string}'."
        
        # Check if intervention parameters need to be updated
        current_intervention_layers = intervention_layers if intervention_layers is not None else app.state.current_intervention_layers
        intervention_params_changed = (
            (requested_intv_weight is not None and requested_intv_weight != app.state.intv_weight) or 
            (current_intervention_layers is not None and current_intervention_layers != app.state.current_intervention_layers)
            or (component_type is not None and component_type != app.state.component_type)
            or (intervention_type is not None and intervention_type != app.state.intervention_type)
        )
        
        # If intervention parameters change, update them if needed
        if intervention_params_changed and app.state.intervention_dir is not None:
            print(f"""Intervention parameters changed: {app.state.intv_weight} -> {requested_intv_weight},
                   {app.state.current_intervention_layers} -> {intervention_layers}, 
                   {app.state.component_type} -> {component_type}, 
                   {app.state.intervention_type} -> {intervention_type}""")
            try:
                args = app.state.args
                
                # Use the provided intervention_layers or fall back to args or current state
                effective_intervention_layers = intervention_layers or args.intervention_layers or app.state.current_intervention_layers
                effective_intervention_type = intervention_type or args.intervention_type
                
                # Prepare intervention components if specified
                if effective_intervention_layers is not None:
                    intervention_start, intervention_end = effective_intervention_layers.split("-")
                    # Select components based on component_type
                    if component_type == "mlp":
                        intervention_components = ["model.layers[{i}].mlp".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
                    elif component_type == "attention":
                        intervention_components = ["model.layers[{i}].self_attn".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
                    else:  # None or invalid value - use both as default
                        intervention_components = ["model.layers[{i}].mlp".format(i=i) for i in range(int(intervention_start), int(intervention_end))] \
                            + ["model.layers[{i}].self_attn".format(i=i) for i in range(int(intervention_start), int(intervention_end))]
                else:
                    intervention_components = None
                    
                # Prepare intervention tokens if step_begin_only is enabled
                if args.step_begin_only:
                    intervention_tokens = [tid for tid in range(app.state.tokenizer.vocab_size) if "\n\n" in app.state.tokenizer.decode(tid)]
                else:
                    intervention_tokens = None
                
                def intervention_fn(model):
                    # Remove existing intervention
                    app.state.intervention_dir.remove_intervention()
                    print(f"Intervention removed with strength {app.state.intv_weight} and layers {app.state.current_intervention_layers}")
                    
                    # Add new intervention if weight is not zero
                    if requested_intv_weight != 0:
                        app.state.intervention_dir.add_intervention(model, requested_intv_weight,
                                                                components=intervention_components,
                                                                type=effective_intervention_type,
                                                                condition_tokens=intervention_tokens,
                                                                normalize_steer_vec=getattr(args, 'normalize_steer_vec', False))
                        print(f"Intervention added with strength {requested_intv_weight} and layers {effective_intervention_layers} and type {effective_intervention_type}")
                
                app.state.engine.engine.model_executor.apply_model(intervention_fn)
                app.state.intv_weight = requested_intv_weight
                app.state.current_intervention_layers = effective_intervention_layers
                app.state.component_type = component_type
                app.state.intervention_type = effective_intervention_type
                # Update the initialized model string to reflect the new intervention weight
                app.state.initialized_model = model_string
                print(f"Updated intervention weight to {requested_intv_weight}, layers to {effective_intervention_layers}, and type to {effective_intervention_type}")
            except Exception as e:
                print(f"Error updating intervention parameters: {str(e)}")
                # Continue anyway since the model is still usable
        
        return True, None
    
    # First time initialization
    try:
        print(f"Initializing model for the first time: {model_string}")
        # Parse the model string
        model_name, intervention_weight = parse_model_name(model_string)
        args = app.state.args
        
        # Check if model_name is valid
        if model_name not in MODELS:
            return False, f"Unknown model name: {model_name}. Available models: {list(MODELS.keys())}"
        
        # Use the provided intervention_layers or fall back to args
        effective_intervention_layers = intervention_layers or args.intervention_layers
        effective_intervention_type = intervention_type or args.intervention_type

        # Initialize the model
        engine, tokenizer, think_start_token_id, think_end_token_id, intervention_dir, head_manager = await initialize_async_llm(
            model_name=model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            with_intervention=intervention_weight,
            intervention_type=effective_intervention_type,
            intervention_direction=args.intervention_direction,
            intervention_layers=effective_intervention_layers,
            step_begin_only=args.step_begin_only,
            disabled_heads_csv=args.disabled_heads_csv,
            head_modify_mode=args.head_modify_mode,
            intv_path=args.intv_path,
            component_type=component_type,
            normalize_steer_vec=getattr(args, 'normalize_steer_vec', False)
        )
        
        # Store in app state
        app.state.engine = engine
        app.state.tokenizer = tokenizer
        app.state.think_start_token_id = think_start_token_id
        app.state.think_end_token_id = think_end_token_id
        app.state.intervention_dir = intervention_dir
        app.state.head_manager = head_manager
        app.state.initialized_model = model_string
        app.state.intv_weight = intervention_weight
        app.state.current_intervention_layers = effective_intervention_layers
        app.state.component_type = component_type
        app.state.intervention_type = effective_intervention_type
        # Pre-compute nowait token IDs if nowait is enabled
        if args.nowait:
            print("Computing nowait token IDs...")
            # Pre-compile regex patterns for better performance
            patterns = [re.compile(rf"^[\.,\s]?{word}[\.,\s]?$") for word in NOWAIT_TARGET_LIST]
            nowait_ids = []
            for token_id in tqdm(range(tokenizer.vocab_size), desc="Nowait suppressed tokens"):
                for pattern in patterns:
                    if pattern.match(tokenizer.decode(token_id).lower()):
                        nowait_ids.append([token_id])
                        break
            app.state.nowait_ids = nowait_ids
            print(f"Nowait suppressed tokens: {[tokenizer.decode(token_id) for token_id in nowait_ids]}")
        else:
            app.state.nowait_ids = None
        
        print(f"AsyncLLMEngine initialized successfully with model: {model_string}")
        return True, None
    except Exception as e:
        error_msg = f"Error initializing model '{model_string}': {str(e)}"
        print(error_msg)
        raise e
        return False, error_msg

@app.get("/")
async def root():
    """Root endpoint for health check."""
    status = {
        "status": "ok",
        "message": "LLM Server is running"
    }
    
    # Add model information if initialized
    if app.state.initialized_model is not None:
        status["model"] = app.state.initialized_model
        status["initialized"] = True
    else:
        status["initialized"] = False
        status["message"] += " (No model initialized yet)"
        
    return status

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions API requests."""
    # Lazy initialize model if needed
    success, error = await lazy_initialize_model(app, request.model,
                                                 request.intervention_layers,
                                                 request.component_type,
                                                 request.intervention_type)
    if not success:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        # Prepare the prompt using the chat template
        prompt = app.state.tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Optionally suppress thinking by appending an empty thinking block
        if getattr(request, "no_think", False):
            prompt = prompt + "\n</think>"
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_completion_tokens or 16384,
            n=request.n
        )
        if app.state.args.nowait and app.state.nowait_ids is not None:
            processor = NoBadWordsLogitsProcessor(app.state.nowait_ids)
            if app.state.args.nowait_str is not None:
                processor._SMALLEST_LOGIT = app.state.args.nowait_str
            sampling_params.logits_processors = [processor]
        
        # Generate completions asynchronously
        request_id = random_uuid()
        results_generator = app.state.engine.generate(prompt, sampling_params, request_id)
        
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_chat_completions(results_generator, request.model),
                media_type="text/event-stream"
            )
        
        # Get final results
        final_output = None
        async for output in results_generator:
            final_output = output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="Failed to generate completions")
        
        # Process outputs
        choices = []
        for i, output in enumerate(final_output.outputs):
            think_length, has_think = get_think_length(
                output.token_ids,
                think_start_id=app.state.think_start_token_id,
                think_end_id=app.state.think_end_token_id,
                max_length=request.max_completion_tokens or 16384
            )
            
            # Handle case where thinking is cut off
            if think_length >= (request.max_completion_tokens or 16384):
                print(f"Thinking length {think_length} is greater than max completion tokens {request.max_completion_tokens or 16384}")
                continue_prompt = "\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
                continue_text = output.text + continue_prompt
                continue_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=MAX_FINISH_TOKENS,
                    n=1
                )
                # Generate continuations asynchronously
                continue_req_id = random_uuid()
                continue_generator = app.state.engine.generate(prompt + continue_text, continue_params, continue_req_id)
                
                # Get final continuation result
                continue_output = None
                async for result in continue_generator:
                    continue_output = result
                
                if continue_output is not None and len(continue_output.outputs) > 0:
                    output.text = continue_text + continue_output.outputs[0].text
                    output.token_ids = output.token_ids + tuple(app.state.tokenizer.encode(continue_prompt)) + tuple(continue_output.outputs[0].token_ids)
            
            try: 
                reasoning, content = output.text.split("</think>")
            except: 
                reasoning, content = "", output.text
            content = content.strip("\n\n")
            choices.append({
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "thinking_length": think_length,
                    "reasoning_content": reasoning,
                    "full_text": output.text
                },
                "prompt_token_ids": final_output.prompt_token_ids,
                "output_token_ids": output.token_ids,
                "finish_reason": "stop"
            })
        
        # Calculate token usage
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
        
        return ChatCompletionResponse(
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

async def stream_chat_completions(results_generator, model_name):
    """Stream chat completions as they are generated."""
    chunk_id = random_uuid()
    
    try:
        async for result in results_generator:
            # Only handle the most recent result
            current_output = result.outputs[0]
            
            # Format response chunk
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant", 
                            "content": current_output.text
                        },
                        "finish_reason": None if not current_output.finished else "stop"
                    }
                ]
            }
            
            # Send the chunk
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # If this is the last chunk, send the [DONE] marker
            if current_output.finished:
                yield "data: [DONE]\n\n"
                break
                
    except Exception as e:
        error_chunk = {
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Handle text completions API requests."""
    # Lazy initialize model if needed
    success, error = await lazy_initialize_model(app, request.model, request.intervention_layers, request.component_type)
    if not success:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 16384,
            n=request.n,
            best_of=request.best_of or request.n
        )
        
        # Optionally suppress thinking by appending an empty thinking block
        prompt_text = request.prompt + ("\n<think>\n</think>" if getattr(request, "no_think", False) else "")
        
        # Generate completions asynchronously
        request_id = random_uuid()
        results_generator = app.state.engine.generate(prompt_text, sampling_params, request_id)
        
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_completions(results_generator, request.model),
                media_type="text/event-stream"
            )
        
        # Get final results
        final_output = None
        async for output in results_generator:
            final_output = output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="Failed to generate completions")
        
        # Process outputs
        choices = []
        for i, output in enumerate(final_output.outputs):
            choices.append({
                "index": i,
                "text": output.text,
                "finish_reason": "stop"
            })
        
        # Calculate token usage
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
        
        return CompletionResponse(
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

async def stream_completions(results_generator, model_name):
    """Stream completions as they are generated."""
    completion_id = random_uuid()
    
    try:
        async for result in results_generator:
            # Only handle the most recent result
            current_output = result.outputs[0]
            
            # Format response chunk
            chunk = {
                "id": completion_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": current_output.text,
                        "finish_reason": None if not current_output.finished else "stop"
                    }
                ]
            }
            
            # Send the chunk
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # If this is the last chunk, send the [DONE] marker
            if current_output.finished:
                yield "data: [DONE]\n\n"
                break
                
    except Exception as e:
        error_chunk = {
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Tokenize a prompt."""
    # Lazy initialize model if needed
    success, error = await lazy_initialize_model(app, request.model, request.intervention_layers, request.component_type)
    if not success:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        tokens = app.state.tokenizer.encode(request.prompt)
        token_strings = [app.state.tokenizer.decode([token]) for token in tokens]
        
        return TokenizeResponse(
            tokens=tokens,
            token_strings=token_strings
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tokenizing prompt: {str(e)}")

# Main function definition
def main():
    """Start the server."""
    global parser
    parser = argparse.ArgumentParser(description="Start an LLM server with OpenAI API compatibility")
    parser.add_argument("--model", type=str, default=None, help="Optional default model name (will be used for command-line initialization)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=18000, help="Maximum model length")
    
    # Intervention arguments
    parser.add_argument("--with_intervention", type=float, default=0.0, help="Intervention strength (0.0 for no intervention)")
    parser.add_argument("--intervention_type", type=str, default="additive", help="Type of intervention (additive, multiplicative, activate, suppress, probe_last_token, probe_last_token_mid_reflect, probe_last_token_temp_<temp>_bias_<bias>, step_confidence, or step_confidence_k_<k_value>)")
    parser.add_argument("--intervention_direction", type=str, default="reflect", help="Direction of intervention")
    parser.add_argument("--intervention_layers", type=str, default=None, help="Layers to apply intervention to (format: start-end)")
    parser.add_argument("--component_type", type=str, default=None, choices=["mlp", "attention"], help="Type of component to apply intervention to (mlp or attention). If not specified, applies to both.")
    parser.add_argument("--intv_path", type=str, default=None, help="Path to intervention direction file (if not specified, uses default path)")
    parser.add_argument("--step_begin_only", action="store_true", help="Apply intervention only at step beginning")
    parser.add_argument("--disabled_heads_csv", type=str, default=None, help="Path to CSV file containing disabled heads information")
    parser.add_argument("--head_modify_mode", type=str, default="disable", choices=["disable", "modify"], help="Mode of head modification")
    parser.add_argument("--nowait", action="store_true", help="Do not use wait in model")
    parser.add_argument("--nowait_str", type=float, default=None, help="Custom value for NoBadWordsLogitsProcessor._SMALLEST_LOGIT")
    parser.add_argument("--normalize_steer_vec", action="store_true", help="Normalize all steering vectors to unit norm before applying interventions")
    # Parse command line arguments
    args = parser.parse_args()
    
    # Store args for use by the app
    if not hasattr(app, '_state'):
        app._state = type('obj', (object,), {})
    app._state.args = args
    
    # Start the server with proper signal handling
    config = uvicorn.Config(app, host=args.host, port=args.port)
    server = uvicorn.Server(config)
    
    # Handle graceful shutdown
    print(f"Starting server on {args.host}:{args.port}, models will be initialized on first request")
    server.run()

if __name__ == "__main__":
    main() 