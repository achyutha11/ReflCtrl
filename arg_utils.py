from argparse import ArgumentParser
from typing import Optional

def add_common_arguments(parser: ArgumentParser,
                        include_mode: bool = False,
                        include_samples: bool = False,
                        include_intervention: bool = False) -> None:
    """
    Add common arguments to an ArgumentParser that are shared across multiple scripts.
    
    Args:
        parser: The ArgumentParser to add arguments to
        include_mode: Whether to include API/offline mode selection
        include_ports: Whether to include server ports configuration
        include_samples: Whether to include number of samples argument
        include_intervention: Whether to include intervention-related arguments
    """
    # Basic arguments used across scripts
    parser.add_argument("--model", type=str, default="deepseek-r1-qwen-1.5b",
                       help="Name of the model to use")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       help="Name of the dataset to process")
    parser.add_argument("--instruction", type=str, default="",
                       help="Additional instruction to append to queries")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of tensor parallel size (default: 1)")
    parser.add_argument("--max_length", type=int, default=8192,
                       help="Maximum length of the generated text (default: 8192)")
    parser.add_argument("--nowait", action="store_true",
                       help="Whether to wait for the server to start (default: False)")
    # Optional argument groups based on script needs
    if include_mode:
        parser.add_argument("--mode", choices=["api", "offline"], default="api",
                          help="Inference mode: 'api' for local server API, 'offline' for batch inference")

    if include_samples:
        parser.add_argument("--n_samples", type=int, default=1,
                          help="Number of samples to generate per question (default: 1)")

    if include_intervention:
        parser.add_argument("--with_intervention", type=float, default=0.0,
                          help="Whether to use intervention (default: 0.0)")
        parser.add_argument("--intervention_type", type=str, default="additive",
                          help="Type of intervention: additive, multiplicative, activate, suppress, probe_last_token, probe_last_token_mid_reflect, probe_last_token_temp_<temp>_bias_<bias>, step_confidence, or step_confidence_k_<k_value> (default: additive)")
        parser.add_argument("--intervention_direction", type=str, default="reflect",
                          help="Direction of intervention (default: reflect)")
        parser.add_argument("--intervention_layers", type=str, default=None,
                          help="Layer range for intervention (e.g. '0-12')")
        parser.add_argument("--step_begin_only", action="store_true",
                          help="Only intervene at the beginning of the step")
        parser.add_argument("--intv_path", type=str, default=None,
                          help="Path to intervention direction file (if not specified, uses default path)")
