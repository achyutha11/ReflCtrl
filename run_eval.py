"""
run_eval.py — Standalone evaluation for ReflCtrl (no server required).

Runs offline vLLM inference with optional intervention hooks, replacing the
server+client architecture (llm_server.py + query_llm.py) for Nautilus.

Modes (set via --intervention_type):
    (no --with_intervention)   baseline, no hooks
    additive                   fixed-lambda steering at every step boundary
    probe_last_token           uncertainty-gated: lambda scales with probe score
    question_adaptive          per-question lambda based on prompt probe score

Usage:
    # Baseline
    python run_eval.py --model deepseek-r1-llama-8b --dataset gsm8k --n_samples 100

    # Fixed steering
    python run_eval.py --model deepseek-r1-llama-8b --dataset gsm8k \\
        --with_intervention -0.48 --intervention_type additive \\
        --intv_path intervention_direction/deepseek-r1-llama-8b/gsm8k/reflect_dir.pt

    # Uncertainty-gated (probe must be trained first via collect_probe.py)
    python run_eval.py --model deepseek-r1-llama-8b --dataset gsm8k \\
        --with_intervention -0.48 --intervention_type probe_last_token \\
        --intv_path intervention_direction/deepseek-r1-llama-8b/gsm8k/reflect_dir.pt \\
        --probe_save_dir probe_data/deepseek-r1-llama-8b/gsm8k/last_token_embedding

    # Question-adaptive (prompt probe must be trained first via train_prompt_probe.py)
    python run_eval.py --model deepseek-r1-llama-8b --dataset gsm8k \\
        --with_intervention -0.39 --intervention_type question_adaptive \\
        --intv_path intervention_direction/deepseek-r1-llama-8b/gsm8k-train/reflect_dir.pt \\
        --probe_save_dir probe_data/deepseek-r1-llama-8b/gsm8k/prompt_probe \\
        --lambda_confident -0.50 --lambda_uncertain -0.20
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import numpy as np
import torch

from arg_utils import add_common_arguments
from hook_utils import InterventionDirection, MODEL_NUM_LAYERS_MAP, ProbeMonitor
from utils import MODELS, analyze_math_results, extract_questions, get_save_dir

DEFAULT_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone ReflCtrl evaluation (offline vLLM).")
    # include_samples adds --n_samples (samples per question for multi-sample eval)
    add_common_arguments(p, include_samples=True, include_intervention=True)
    # Override defaults: use correct DeepSeek-R1 instruction and step-boundary-only steering
    p.set_defaults(
        instruction=DEFAULT_INSTRUCTION,
        step_begin_only=True,   # always steer only at \n\n boundaries, not every token
    )
    p.add_argument("--n_questions", type=int, default=None,
                   help="Limit to first N questions (default: all).")
    p.add_argument("--probe_save_dir", type=str, default=None,
                   help="Dir with clf_weights.pt / clf_bias.pt for probe_last_token mode.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: derived from args).")
    # Question-adaptive args
    p.add_argument("--lambda_confident", type=float, default=-0.50,
                   help="Lambda for confident questions (question_adaptive mode).")
    p.add_argument("--lambda_uncertain", type=float, default=-0.20,
                   help="Lambda for uncertain questions (question_adaptive mode).")
    p.add_argument("--adaptive_threshold", type=float, default=0.0,
                   help="Probe score threshold: >threshold=confident (question_adaptive mode).")
    return p.parse_args()


def build_layer_range(model_name: str, intervention_layers: str | None) -> list[int]:
    """Parse --intervention_layers 'a-b' into a list of layer indices."""
    n = MODEL_NUM_LAYERS_MAP[model_name]
    if intervention_layers is None:
        # Default: skip first 6 and last 6 layers (as in the paper)
        return list(range(6, n - 6))
    lo, hi = intervention_layers.split("-")
    return list(range(int(lo), int(hi) + 1))


def run_eval(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Load tokenizer (for prompt building — separate from vLLM's internal one)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], trust_remote_code=True)

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    questions = extract_questions(args.dataset)
    if args.n_questions is not None:
        questions = questions[:args.n_questions]
    print(f"Evaluating on {len(questions)} questions from {args.dataset}")

    # Build prompts using chat template (correct for DeepSeek-R1 family)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q + args.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    # ------------------------------------------------------------------
    # Initialise vLLM
    # ------------------------------------------------------------------
    llm = LLM(
        model=MODELS[args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_length + 2048,
        enforce_eager=True,  # required for forward hooks
    )

    # ------------------------------------------------------------------
    # Register intervention hooks (optional)
    # ------------------------------------------------------------------
    weight_manager = None
    if args.with_intervention != 0.0:
        if args.intv_path is None:
            raise ValueError("--intv_path required when --with_intervention != 0.")
        intv_dir = InterventionDirection.load(args.intv_path)

        # Determine which layer components to steer
        active_layers = build_layer_range(args.model, args.intervention_layers)
        active_components = [
            c for c in intv_dir.components
            if any(f"layers[{i}]" in c for i in active_layers)
        ]

        # Step-boundary-only: collect token IDs that encode "\n\n".
        # ConditionalInterventionHook wraps every layer hook so it only fires
        # when the current input token is one of these delimiters — i.e. when
        # the model is about to generate the first token of a new reasoning step.
        # This avoids mid-sentence activation disruption.
        condition_tokens = None
        if args.step_begin_only:
            tok = llm.get_tokenizer()
            condition_tokens = [
                tid for tid in range(tok.vocab_size)
                if "\n\n" in tok.decode(tid)
            ]
            print(f"Step-boundary-only steering: {len(condition_tokens)} delimiter token(s)")

        def _apply_intervention(model):
            nonlocal weight_manager
            # question_adaptive uses additive hooks; lambda is updated per-question
            hook_type = "additive" if args.intervention_type == "question_adaptive" else args.intervention_type
            weight_manager = intv_dir.add_intervention(
                model,
                weight=args.with_intervention,
                type=hook_type,
                probe_save_dir=args.probe_save_dir,
                components=active_components,
                condition_tokens=condition_tokens,
            )

        llm.apply_model(_apply_intervention)
        print(
            f"Intervention: type={args.intervention_type}, "
            f"weight={args.with_intervention}, "
            f"step_begin_only={args.step_begin_only}, "
            f"layers={args.intervention_layers or 'default (skip 6/6)'}"
        )
    else:
        print("Baseline: no intervention.")

    # ------------------------------------------------------------------
    # Sampling parameters
    # ------------------------------------------------------------------
    tokenizer_vllm = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.6 if args.n_samples > 1 else 0.0,
        top_p=0.95,
        max_tokens=args.max_length,
    )

    # ------------------------------------------------------------------
    # Question-adaptive: set up prompt probe for per-question scoring
    # ------------------------------------------------------------------
    prompt_probe = None
    prompt_probe_hooks = []
    prompt_proj_buffer = {}
    if args.intervention_type == "question_adaptive":
        if args.probe_save_dir is None:
            raise ValueError("--probe_save_dir required for question_adaptive mode.")
        prompt_probe = ProbeMonitor(args.probe_save_dir)

        # Register hooks to capture direction projections on the prompt
        component_names = sorted(intv_dir.components.keys())
        direction_vectors = {
            comp: (
                intv_dir.components[comp].mean_diff
                / intv_dir.components[comp].mean_diff.norm()
            ).float()
            for comp in component_names
        }

        def make_proj_hook(comp_name, direction):
            def hook(module, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.dim() == 3:
                    last = hidden[:, -1, :].float()
                else:
                    last = hidden[-1, :].float().unsqueeze(0)
                proj = (last @ direction.to(last.device)).squeeze()
                prompt_proj_buffer[comp_name] = proj.item()
            return hook

        def register_proj_hooks(model):
            for comp_name, direction in direction_vectors.items():
                module = eval(f"model.{comp_name}")
                handle = module.register_forward_hook(make_proj_hook(comp_name, direction))
                prompt_probe_hooks.append(handle)

        llm.apply_model(register_proj_hooks)
        print(f"Question-adaptive: lambda_confident={args.lambda_confident}, "
              f"lambda_uncertain={args.lambda_uncertain}, "
              f"threshold={args.adaptive_threshold}")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    if args.intervention_type == "question_adaptive":
        # Per-question generation: score prompt, set lambda, generate
        print("Running question-adaptive inference (one question at a time)...")
        outputs = []
        score_sp = SamplingParams(max_tokens=1, temperature=0.0)
        lambda_choices = []

        # Collect all intervention hooks so we can update their weights
        intervention_hooks = []
        def collect_hooks(model):
            for comp in active_components:
                module = eval(f"model.{comp}")
                for h in module._forward_hooks.values():
                    # Find the ConditionalInterventionHook wrapping a LinearInterventionHook
                    from hook_utils import ConditionalInterventionHook, LinearInterventionHook
                    if isinstance(h, ConditionalInterventionHook):
                        if isinstance(h.hook, LinearInterventionHook):
                            intervention_hooks.append(h.hook)
                    elif isinstance(h, LinearInterventionHook):
                        intervention_hooks.append(h)
        llm.apply_model(collect_hooks)
        print(f"  Found {len(intervention_hooks)} intervention hooks to update")

        for i, prompt in enumerate(tqdm(prompts, desc="Question-adaptive eval")):
            # 1) Score the prompt (disable steering during scoring pass)
            for hook in intervention_hooks:
                hook.weight = 0.0
            prompt_proj_buffer.clear()
            llm.generate(prompt, score_sp)

            # Build feature vector and get probe score
            component_names_sorted = sorted(prompt_proj_buffer.keys())
            features = torch.tensor([[prompt_proj_buffer[c] for c in component_names_sorted]])
            _, score = prompt_probe.predict(features)
            score_val = score.item()

            # 2) Map score to lambda
            if score_val > args.adaptive_threshold:
                chosen_lambda = args.lambda_confident
            else:
                chosen_lambda = args.lambda_uncertain
            lambda_choices.append(chosen_lambda)

            # 3) Update all hook weights
            for hook in intervention_hooks:
                hook.weight = chosen_lambda

            # 4) Generate
            out = llm.generate(prompt, sampling_params)
            outputs.extend(out)

            if (i + 1) % 50 == 0:
                n_conf = sum(1 for l in lambda_choices if l == args.lambda_confident)
                print(f"  [{i+1}/{len(prompts)}] confident: {n_conf}, "
                      f"uncertain: {i+1-n_conf}")

        # Remove projection hooks
        for h in prompt_probe_hooks:
            h.remove()

        # Print adaptive stats
        n_conf = sum(1 for l in lambda_choices if l == args.lambda_confident)
        print(f"\nAdaptive stats: {n_conf}/{len(lambda_choices)} confident "
              f"({100*n_conf/len(lambda_choices):.1f}%), "
              f"{len(lambda_choices)-n_conf} uncertain")
    else:
        print("Running inference...")
        outputs = llm.generate(prompts, sampling_params)

    # ------------------------------------------------------------------
    # Parse outputs into the format expected by analyze_math_results
    # ------------------------------------------------------------------
    # analyze_math_results expects: List[List[Dict]] where the outer list
    # is indexed by sample and the inner list contains one dict per question.
    print("Parsing outputs...")
    responses_by_sample: list[list[dict]] = [[] for _ in range(args.n_samples)]
    for output in tqdm(outputs, desc="Parsing outputs"):
        for s, completion in enumerate(output.outputs):
            text = completion.text

            # Separate <think> content from final answer.
            think_marker = "</think>"
            if think_marker in text:
                think_part, answer_part = text.split(think_marker, 1)
            else:
                think_part, answer_part = text, ""
            # Strip leading <think> tag if present
            think_part = think_part.strip()
            if think_part.startswith("<think>"):
                think_part = think_part[len("<think>"):].strip()

            # Count thinking length in tokens
            think_token_ids = tokenizer_vllm.encode(think_part, add_special_tokens=False)
            thinking_length = len(think_token_ids)

            responses_by_sample[s].append({
                "content": answer_part.strip(),
                "reasoning": think_part,
                "thinking_length": thinking_length,
            })

    # ------------------------------------------------------------------
    # Score results
    # ------------------------------------------------------------------
    print("Scoring results...")
    aggregate_stats, analyzed_results = analyze_math_results(responses_by_sample, args.dataset)

    print("\n=== Results ===")
    print(f"  accuracy:           {aggregate_stats['accuracy']:.4f}")
    print(f"  avg_thinking_length:{aggregate_stats['avg_thinking_length']:.1f} tokens")
    print(f"  n_questions:        {len(questions)}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        is_steered = (args.with_intervention != 0.0)
        out_dir = get_save_dir(
            dataset=args.dataset,
            model=args.model,
            instruction=args.instruction,
            with_intervention=args.with_intervention,
            intervention_direction=args.intervention_direction if is_steered else None,
            intervention_layers=args.intervention_layers if is_steered else None,
            step_begin_only=args.step_begin_only if is_steered else False,
            intervention_type=args.intervention_type if is_steered else "additive",
            nowait=args.nowait,
            intv_path=args.intv_path if is_steered else None,
        )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"results_samples{args.n_samples}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "aggregate": aggregate_stats,
                "args": vars(args),
                "questions": questions,
                "sample_results": analyzed_results["sample_results"],
                "answers": analyzed_results["answers"],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
