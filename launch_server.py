#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional
import uuid
import GPUtil
import logging
import logging.handlers
from datetime import datetime

# Import uvicorn and FastAPI frameworks
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx
from contextlib import asynccontextmanager

# Configure logging
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Set up logging to file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"server_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=15)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Server monitoring and load balancing
class ServerStatus:
    def __init__(self, host: str, port: int, gpu_id: int, log_file: str = None):
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.url = f"http://{host}:{port}"
        self.health_url = f"{self.url}/"
        self.active_requests = 0
        self.total_requests = 0
        self.last_health_check = 0
        self.is_healthy = False
        self.model = None

class LoadBalancer:
    def __init__(self):
        self.servers: List[ServerStatus] = []
        self.health_check_interval = 5  # seconds
        self.last_health_check = 0
    
    def add_server(self, host: str, port: int, gpu_id: int, log_file: str = None):
        server = ServerStatus(host, port, gpu_id, log_file)
        self.servers.append(server)
        logging.info(f"Added server to load balancer: {host}:{port} on GPU {gpu_id}")
        return server
    
    def get_server(self, preferred_gpu: Optional[int] = None) -> Optional[ServerStatus]:
        """
        Get the next available server using a load balancing algorithm.
        If preferred_gpu is specified, attempt to use a server on that GPU.
        """
        # Filter for healthy servers
        healthy_servers = [s for s in self.servers if s.is_healthy]
        logging.debug(f"Total servers: {len(self.servers)}, Healthy servers: {len(healthy_servers)}")
        if not healthy_servers:
            return None
        
        # If preferred GPU is specified and available
        if preferred_gpu is not None:
            gpu_servers = [s for s in healthy_servers if s.gpu_id == preferred_gpu]
            if gpu_servers:
                # Use least active connections for this GPU
                return min(gpu_servers, key=lambda s: s.active_requests)
        
        # Otherwise use least active connections across all servers
        return min(healthy_servers, key=lambda s: s.active_requests)
    
    async def check_health(self):
        """Check the health of all servers."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        check_tasks = []
        
        logging.debug(f"Starting health check for {len(self.servers)} servers")
        async with httpx.AsyncClient(timeout=2.0) as client:
            for server in self.servers:
                check_tasks.append(self.check_server_health(client, server))
            
            # Wait for all health checks to complete
            await asyncio.gather(*check_tasks, return_exceptions=True)
        logging.debug(f"Health check complete. Healthy servers: {sum(1 for s in self.servers if s.is_healthy)}")
    
    async def check_server_health(self, client: httpx.AsyncClient, server: ServerStatus):
        """Check health of an individual server."""
        was_healthy = server.is_healthy
        try:
            response = await client.get(server.health_url)
            if response.status_code == 200:
                data = response.json()
                server.is_healthy = True
                server.model = data.get("model")
                server.last_health_check = time.time()
                if not was_healthy:
                    logging.info(f"Server {server.host}:{server.port} on GPU {server.gpu_id} is now healthy")
                return
        except Exception as e:
            logging.debug(f"Health check failed for server {server.host}:{server.port}: {str(e)}")
        
        # Mark as unhealthy if request failed
        if was_healthy:
            logging.warning(f"Server {server.host}:{server.port} on GPU {server.gpu_id} is now unhealthy")
        server.is_healthy = False

# Initialize FastAPI app for the router
@asynccontextmanager
async def lifespan(app):
    # Only create a new load balancer if one doesn't already exist
    if not hasattr(app.state, 'load_balancer'):
        logging.info("Creating new LoadBalancer in lifespan")
        app.state.load_balancer = LoadBalancer()
    else:
        logging.info(f"Using existing LoadBalancer with {len(app.state.load_balancer.servers)} servers")
    
    app.state.health_check_task = None
    
    # Start health check background task
    async def health_check_loop():
        while True:
            try:
                await app.state.load_balancer.check_health()
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(5)  # Wait longer after an error
    
    logging.info("Starting health check background task")
    app.state.health_check_task = asyncio.create_task(health_check_loop())
    
    yield
    
    # Cleanup
    logging.info("Shutting down health check task")
    if app.state.health_check_task:
        app.state.health_check_task.cancel()
        try:
            await app.state.health_check_task
        except asyncio.CancelledError:
            pass
        logging.info("Health check task shut down successfully")

app = FastAPI(title="LLM Router with Load Balancing", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models (same as in llm_server.py)
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
    preferred_gpu: Optional[int] = None  # New field to allow request routing to specific GPU
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None
    intervention_type: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    n: int = 1
    best_of: Optional[int] = None
    stream: bool = False
    preferred_gpu: Optional[int] = None  # New field to allow request routing to specific GPU
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None

class TokenizeRequest(BaseModel):
    model: str
    prompt: str
    preferred_gpu: Optional[int] = None  # New field to allow request routing to specific GPU
    intervention_layers: Optional[str] = None
    component_type: Optional[str] = None

# Router endpoints
@app.get("/")
async def router_status():
    """Get status of the router and all servers."""
    lb = app.state.load_balancer
    
    # Force a health check
    await lb.check_health()
    
    servers_info = []
    for server in lb.servers:
        server_info = {
            "url": server.url,
            "gpu_id": server.gpu_id,
            "healthy": server.is_healthy,
            "active_requests": server.active_requests,
            "total_requests": server.total_requests,
            "model": server.model,
            "log_file": server.log_file
        }
        servers_info.append(server_info)
    
    logging.debug(f"Status request: {len(lb.servers)} servers, {sum(1 for s in lb.servers if s.is_healthy)} healthy")
    
    return {
        "status": "ok",
        "servers": servers_info,
        "total_servers": len(lb.servers),
        "healthy_servers": sum(1 for s in lb.servers if s.is_healthy)
    }

@app.post("/v1/chat/completions")
async def router_chat_completions(request: ChatCompletionRequest):
    """Route chat completions to an available server."""
    request_id = str(uuid.uuid4())
    # Log full request details
    logging.info(f"[{request_id}] Received chat completion request:\n"
                f"Model: {request.model}\n"
                f"Temperature: {request.temperature}\n"
                f"Top_p: {request.top_p}\n"
                f"Stream: {request.stream}\n"
                f"Max completion tokens: {request.max_completion_tokens}\n"
                f"Intervention layers: {request.intervention_layers}\n"
                f"Component type: {request.component_type}\n"
                f"Intervention type: {request.intervention_type}")
    
    # Get the most available server
    server = app.state.load_balancer.get_server(request.preferred_gpu)
    if not server:
        logging.warning(f"[{request_id}] No healthy servers available for chat completion request")
        raise HTTPException(status_code=503, detail="No healthy servers available")
    
    # Clean up our custom fields before forwarding
    request_dict = request.model_dump()
    if "preferred_gpu" in request_dict:
        del request_dict["preferred_gpu"]
    
    # Forward the request
    server.active_requests += 1
    server.total_requests += 1
    logging.info(f"[{request_id}] Routing chat completion to server {server.host}:{server.port} (GPU {server.gpu_id}) - Active requests: {server.active_requests}")
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if request.stream:
                logging.info(f"[{request_id}] Starting streaming response from server {server.url}")
                return StreamingResponse(
                    stream_from_server(client, f"{server.url}/v1/chat/completions", request_dict, server, request_id),
                    media_type="text/event-stream"
                )
            else:
                start_time = time.time()
                response = await client.post(
                    f"{server.url}/v1/chat/completions",
                    json=request_dict
                )
                
                if response.status_code != 200:
                    logging.error(f"[{request_id}] Error from server {server.url}: {response.status_code} {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                
                server.active_requests -= 1
                elapsed_time = time.time() - start_time
                response_data = response.json()
                logging.info(f"[{request_id}] Completed chat completion from {server.url} in {elapsed_time:.2f}s\n"
                           f"Response: {json.dumps(response_data, indent=2)}")
                return response_data
    except Exception as e:
        server.active_requests -= 1
        logging.error(f"[{request_id}] Error routing request to {server.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM server: {str(e)}")

@app.post("/v1/completions")
async def router_completions(request: CompletionRequest):
    """Route text completions to an available server."""
    request_id = str(uuid.uuid4())
    # Log full request details
    logging.info(f"[{request_id}] Received completion request:\n"
                f"Model: {request.model}\n"
                f"Temperature: {request.temperature}\n"
                f"Top_p: {request.top_p}\n"
                f"Stream: {request.stream}\n"
                f"Max tokens: {request.max_tokens}\n"
                f"N: {request.n}\n"
                f"Best of: {request.best_of}\n"
                f"Intervention layers: {request.intervention_layers}\n"
                f"Component type: {request.component_type}\n"
                f"Prompt: {request.prompt}")
    
    # Get the most available server
    server = app.state.load_balancer.get_server(request.preferred_gpu)
    if not server:
        logging.warning(f"[{request_id}] No healthy servers available for completion request")
        raise HTTPException(status_code=503, detail="No healthy servers available")
    
    # Clean up our custom fields before forwarding
    request_dict = request.model_dump()
    if "preferred_gpu" in request_dict:
        del request_dict["preferred_gpu"]
    
    # Forward the request
    server.active_requests += 1
    server.total_requests += 1
    logging.info(f"[{request_id}] Routing completion to server {server.host}:{server.port} (GPU {server.gpu_id}) - Active requests: {server.active_requests}")
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if request.stream:
                logging.info(f"[{request_id}] Starting streaming response from server {server.url}")
                return StreamingResponse(
                    stream_from_server(client, f"{server.url}/v1/completions", request_dict, server, request_id),
                    media_type="text/event-stream"
                )
            else:
                start_time = time.time()
                response = await client.post(
                    f"{server.url}/v1/completions",
                    json=request_dict
                )
                server.active_requests -= 1
                
                if response.status_code != 200:
                    logging.error(f"[{request_id}] Error from server {server.url}: {response.status_code} {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                
                elapsed_time = time.time() - start_time
                response_data = response.json()
                logging.info(f"[{request_id}] Completed completion request from {server.url} in {elapsed_time:.2f}s\n"
                           f"Response: {json.dumps(response_data, indent=2)}")
                return response_data
    except Exception as e:
        server.active_requests -= 1
        logging.error(f"[{request_id}] Error routing request to {server.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM server: {str(e)}")

@app.post("/v1/tokenize")
async def router_tokenize(request: TokenizeRequest):
    """Route tokenize requests to an available server."""
    request_id = str(uuid.uuid4())
    # Log full request details
    logging.info(f"[{request_id}] Received tokenize request:\n"
                f"Model: {request.model}\n"
                f"Intervention layers: {request.intervention_layers}\n"
                f"Component type: {request.component_type}\n"
                f"Prompt: {request.prompt}")
    
    # Get the most available server
    server = app.state.load_balancer.get_server(request.preferred_gpu)
    if not server:
        logging.warning(f"[{request_id}] No healthy servers available for tokenize request")
        raise HTTPException(status_code=503, detail="No healthy servers available")
    
    # Clean up our custom fields before forwarding
    request_dict = request.model_dump()
    if "preferred_gpu" in request_dict:
        del request_dict["preferred_gpu"]
    
    # Forward the request
    server.active_requests += 1
    server.total_requests += 1
    logging.info(f"[{request_id}] Routing tokenize request to server {server.host}:{server.port} (GPU {server.gpu_id}) - Active requests: {server.active_requests}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            response = await client.post(
                f"{server.url}/v1/tokenize",
                json=request_dict
            )
            server.active_requests -= 1
            
            if response.status_code != 200:
                logging.error(f"[{request_id}] Error from server {server.url}: {response.status_code} {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            elapsed_time = time.time() - start_time
            response_data = response.json()
            logging.info(f"[{request_id}] Completed tokenize request from {server.url} in {elapsed_time:.2f}s\n"
                       f"Response: {json.dumps(response_data, indent=2)}")
            return response_data
    except Exception as e:
        server.active_requests -= 1
        logging.error(f"[{request_id}] Error routing request to {server.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM server: {str(e)}")

async def stream_from_server(client, endpoint, request_data, server, request_id):
    """Stream response from backend server."""
    try:
        logging.debug(f"[{request_id}] Starting stream from {endpoint}")
        async with client.stream("POST", endpoint, json=request_data) as response:
            async for chunk in response.aiter_text():
                # Log each chunk of the streaming response
                logging.debug(f"[{request_id}] Stream chunk: {chunk}")
                yield chunk
        logging.debug(f"[{request_id}] Stream completed from {endpoint}")
    except Exception as e:
        logging.error(f"[{request_id}] Error streaming from {endpoint}: {str(e)}")
        raise
    finally:
        server.active_requests -= 1
        logging.debug(f"[{request_id}] Reduced active requests for server {server.url} to {server.active_requests}")

@app.get("/v1/args")
async def get_launch_args():
    """Get the command line arguments used to launch the servers."""
    if not hasattr(app.state, 'launch_args'):
        raise HTTPException(status_code=404, detail="Launch arguments not found")
    
    # Convert any non-JSON serializable values to strings
    args_dict = {}
    for key, value in app.state.launch_args.items():
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            args_dict[key] = value
        else:
            args_dict[key] = str(value)
    
    return {
        "status": "ok",
        "launch_arguments": args_dict
    }

# GPU Management functions
def get_available_gpus():
    """Get a list of available GPUs."""
    try:
        gpus = GPUtil.getGPUs()
        logging.info(f"Detected {len(gpus)} GPUs: {[gpu.id for gpu in gpus]}")
        return gpus
    except Exception as e:
        logging.error(f"Unable to get GPU information: {str(e)}")
        return []

def launch_server(gpu_id, model, port, host="0.0.0.0", tensor_parallel_size=1, max_model_len=32768, 
                 intervention_weight=0.0, intervention_type="additive", 
                 intervention_direction="reflect", intervention_layers=None, 
                 step_begin_only=False, disabled_heads_csv=None, head_modify_mode="disable",
                 nowait=False, intv_path=None, component_type=None, nowait_str=None, normalize_steer_vec=False):
    """Launch a vLLM server on a specific GPU."""
    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    server_script = "llm_server.py"
    logging.info("Using vLLM backend")
    
    # Build the base command
    cmd = [
        "python", server_script,
        "--model", model,
        "--port", str(port),
        "--host", host,
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--max_model_len", str(max_model_len),
        "--with_intervention", str(intervention_weight),
        "--intervention_type", intervention_type,
        "--intervention_direction", intervention_direction
    ]
    if nowait:
        cmd.append("--nowait")
    if nowait_str is not None:
        cmd.extend(["--nowait_str", str(nowait_str)])
    if normalize_steer_vec:
        cmd.append("--normalize_steer_vec")
    
    # Add optional arguments
    if intervention_layers:
        cmd.extend(["--intervention_layers", intervention_layers])
    if step_begin_only:
        cmd.append("--step_begin_only")
    if disabled_heads_csv:
        cmd.extend(["--disabled_heads_csv", disabled_heads_csv])
    if head_modify_mode:
        cmd.extend(["--head_modify_mode", head_modify_mode])
    if intv_path:
        cmd.extend(["--intv_path", intv_path])
    if component_type:
        cmd.extend(["--component_type", component_type])
    # Create a log file for this server
    log_dir = "logs/servers"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"gpu{gpu_id}_port{port}_vllm.log")
    log_fh = open(log_file, 'w')
    
    # Launch the server process with logging
    logging.info(f"Launching server on GPU {gpu_id} at port {port} with command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=log_fh, 
        stderr=subprocess.STDOUT,
        bufsize=1
    )
    
    # Store log file handle with process for cleanup
    proc.log_file = log_file
    proc.log_handle = log_fh
    
    logging.info(f"Server logs for GPU {gpu_id} will be saved to {log_file}")
    return proc

def main():
    """Main entry point for the load balancing router."""
    parser = argparse.ArgumentParser(
        description="Launch multiple LLM servers across GPUs with load balancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python launch_server.py --gpus 0,1 --model deepseek-r1-llama-8b
        """
    )
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--model", type=str, required=True, help="Model name to load")
    parser.add_argument("--router_port", type=int, default=8000, help="Port for the router to listen on")
    parser.add_argument("--server_port_start", type=int, default=8100, help="Starting port for the LLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for each server")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length")
    parser.add_argument("--with_intervention", type=float, default=0.0, help="Intervention strength (0.0 for no intervention)")
    parser.add_argument("--intervention_type", type=str, default="additive", help="Type of intervention (additive, multiplicative, activate, suppress, probe_last_token, probe_last_token_mid_reflect, probe_last_token_temp_<temp>_bias_<bias>, step_confidence, or step_confidence_k_<k_value>)")
    parser.add_argument("--intervention_direction", type=str, default="reflect", help="Direction of intervention")
    parser.add_argument("--intervention_layers", type=str, default=None, help="Layers to apply intervention to (format: start-end)")
    parser.add_argument("--component_type", type=str, default=None, choices=["mlp", "attention"], help="Type of component to apply intervention to (mlp or attention). If not specified, applies to both.")
    parser.add_argument("--intv_path", type=str, default=None, help="Path to intervention direction file (if not specified, uses default path)")
    parser.add_argument("--step_begin_only", action="store_true", help="Apply intervention only at step beginning")
    parser.add_argument("--disabled_heads_csv", type=str, default=None, help="CSV file containing disabled heads")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      help="Logging level")
    parser.add_argument("--head_modify_mode", type=str, default="disable", choices=["disable", "modify"], help="Mode of head modification")
    parser.add_argument("--nowait", action="store_true", help="Do not use wait in model")
    parser.add_argument("--nowait_str", type=float, default=None, help="Custom value for NoBadWordsLogitsProcessor._SMALLEST_LOGIT")
    parser.add_argument("--normalize_steer_vec", action="store_true", help="Normalize all steering vectors to unit norm before applying interventions")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    logging.info(f"Starting LLM Router with log level {args.log_level}")
    
    # Parse GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        logging.error("No valid GPU IDs provided. Exiting.")
        return
    
    logging.info(f"Requested GPUs: {gpu_ids}")
    logging.info(f"Model: {args.model}")
    
    # Check available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        logging.error("No GPUs detected. Exiting.")
        return
    
    available_gpu_ids = [gpu.id for gpu in available_gpus]
    for gpu_id in gpu_ids:
        if gpu_id not in available_gpu_ids:
            logging.warning(f"GPU {gpu_id} not found in available GPUs {available_gpu_ids}")
    
    # Store args for use by the app
    if not hasattr(app.state, 'load_balancer'):
        app.state.load_balancer = LoadBalancer()
        logging.info(f"Created load balancer")
    
    # Store command line arguments in app state
    app.state.launch_args = vars(args)
    logging.info("Stored command line arguments in app state")
    
    # Launch servers on each GPU
    server_processes = []
    server_ports = []
    
    for i, gpu_id in enumerate(gpu_ids):
        port = args.server_port_start + i
        server_ports.append(port)
        
        proc = launch_server(
            gpu_id=gpu_id,
            model=args.model,
            port=port,
            host=args.host,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            intervention_weight=args.with_intervention,
            intervention_type=args.intervention_type,
            intervention_direction=args.intervention_direction,
            intervention_layers=args.intervention_layers,
            step_begin_only=args.step_begin_only,
            disabled_heads_csv=args.disabled_heads_csv,
            head_modify_mode=args.head_modify_mode,
            nowait=args.nowait,
            intv_path=args.intv_path,
            component_type=args.component_type,
            nowait_str=args.nowait_str,
            normalize_steer_vec=args.normalize_steer_vec
        )
        
        server_processes.append(proc)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logging.info("Shutting down servers...")
        for proc in server_processes:
            if proc.poll() is None:  # If process is still running
                proc.terminate()
        
        # Wait for all processes to terminate
        for proc in server_processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning(f"Process didn't terminate gracefully, killing...")
                proc.kill()
            
            # Close log file handles
            if hasattr(proc, 'log_handle') and proc.log_handle:
                proc.log_handle.close()
        
        logging.info("All servers shut down.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Store server info for the router
    for i, (gpu_id, port) in enumerate(zip(gpu_ids, server_ports)):
        # Add to app state
        if not hasattr(app.state, 'load_balancer'):
            # Create it if we're running outside the FastAPI context
            app.state.load_balancer = LoadBalancer()
            logging.info(f"Created load balancer")
        
        # Get the process and log file info
        proc = server_processes[i]
        log_file = proc.log_file if hasattr(proc, 'log_file') else None
        
        logging.info(f"Adding server {args.host}:{port} to load balancer")
        app.state.load_balancer.add_server(args.host, port, gpu_id, log_file)
        logging.info(f"Current servers in load balancer: {[(s.host, s.port, s.gpu_id) for s in app.state.load_balancer.servers]}")
    
    # Start the router
    logging.info(f"Starting router on {args.host}:{args.router_port}")
    
    # Configure and start the uvicorn server with logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.router_port,
        log_level=args.log_level.lower(),
        log_config=log_config
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
