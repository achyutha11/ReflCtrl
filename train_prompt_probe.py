"""
train_prompt_probe.py — Train probe on prompt-last-token activations.

Unlike train_probe.py (which uses </think> token activations), this probe
uses the model's activations at the last token of the prompt — before any
generation happens. This enables question-level difficulty classification
for adaptive lambda selection.

The feature vector is the same format: [cos(d_l, z_l)] for all layers × modules,
giving a 64-dim vector for the 32-layer Llama-8B.

Usage:
    python train_prompt_probe.py \
        --model deepseek-r1-llama-8b \
        --dataset gsm8k \
        --results_path data/gsm8k/.../results_samples1.json \
        --intv_path intervention_direction/deepseek-r1-llama-8b/gsm8k-train/reflect_dir.pt \
        --probe_save_dir probe_data/deepseek-r1-llama-8b/gsm8k/prompt_probe
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from vllm import LLM, SamplingParams

from hook_utils import InterventionDirection
from utils import MODELS, extract_questions

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train prompt-level uncertainty probe.")
    p.add_argument("--model", type=str, default="deepseek-r1-llama-8b")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--results_path", type=str, required=True,
                   help="Path to results_samples1.json from run_eval.py baseline run.")
    p.add_argument("--intv_path", type=str, required=True,
                   help="Path to reflect_dir.pt (InterventionDirection).")
    p.add_argument("--probe_save_dir", type=str,
                   default="probe_data/deepseek-r1-llama-8b/gsm8k/prompt_probe",
                   help="Directory to save clf_weights.pt and clf_bias.pt.")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)
    return p.parse_args()


def collect_prompt_projections(
    llm: LLM,
    questions: list[str],
    correctness: list[bool],
    instruction: str,
    intv_dir: InterventionDirection,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each question, feed only the prompt through the model and capture
    the direction projections at the last prompt token.

    Returns:
        X: [n_examples, n_components], y: [n_examples]
    """
    tokenizer = llm.get_tokenizer()

    component_names = sorted(intv_dir.components.keys())
    direction_vectors = {
        comp: (
            intv_dir.components[comp].mean_diff
            / intv_dir.components[comp].mean_diff.norm()
        ).float()
        for comp in component_names
    }

    proj_buffer: dict[str, float] = {}
    handle_refs: list = []

    def make_hook(comp_name: str, direction: torch.Tensor):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                last = hidden[:, -1, :].float()
            else:
                last = hidden[-1, :].float().unsqueeze(0)
            proj = (last @ direction.to(last.device)).squeeze()
            proj_buffer[comp_name] = proj.item()
        return hook

    def register_all_hooks(model):
        for comp_name, direction in direction_vectors.items():
            module = eval(f"model.{comp_name}")
            handle = module.register_forward_hook(make_hook(comp_name, direction))
            handle_refs.append(handle)

    llm.apply_model(register_all_hooks)

    # Generate 1 token just to trigger forward pass on the prompt
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    X_list, y_list = [], []

    for i, (question, correct) in enumerate(zip(questions, correctness)):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        proj_buffer.clear()
        llm.generate(prompt, sp)

        if len(proj_buffer) < len(component_names):
            continue

        feature = [proj_buffer[c] for c in component_names]
        X_list.append(feature)
        y_list.append(int(correct))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] {len(X_list)} examples collected")

    for h in handle_refs:
        h.remove()

    print(f"\nTotal: {len(X_list)} usable examples.")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_and_save_probe(X: np.ndarray, y: np.ndarray, save_dir: str) -> None:
    """Train logistic regression and save weights."""
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need both correct and incorrect examples. Got: {np.bincount(y)}")

    print(f"Feature vector size: {X.shape[1]} ({X.shape[1]//2} layers x 2 components)")
    print(f"Class balance: {y.sum()} correct, {(1-y).sum()} incorrect out of {len(y)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)

    val_scores = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_scores)
    val_acc = clf.score(X_val, y_val)
    print(f"Prompt probe val accuracy: {val_acc:.4f}  |  AUROC: {auroc:.4f}")

    os.makedirs(save_dir, exist_ok=True)

    weights = clf.coef_[0].astype(np.float32)
    bias = clf.intercept_.astype(np.float32)

    torch.save(weights, os.path.join(save_dir, "clf_weights.pt"))
    torch.save(bias, os.path.join(save_dir, "clf_bias.pt"))

    meta = {
        "val_accuracy": float(val_acc),
        "auroc": float(auroc),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "feature_dim": int(X.shape[1]),
        "feature_type": "prompt_direction_projection",
    }
    with open(os.path.join(save_dir, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPrompt probe saved to {save_dir}/")


def main():
    args = parse_args()

    print(f"Loading results from {args.results_path}")
    with open(args.results_path) as f:
        saved = json.load(f)

    questions = saved.get("questions")
    sample_results_list = saved.get("sample_results", [])
    if not sample_results_list:
        raise ValueError("No sample_results found in results file.")

    sample_results = sample_results_list[0]
    correctness = sample_results["correctness"]

    if questions is None:
        questions = extract_questions(args.dataset)
        questions = questions[:len(correctness)]

    print(f"Loaded {len(questions)} questions  |  "
          f"correct: {sum(correctness)}/{len(correctness)}")

    print(f"\nLoading direction vectors from {args.intv_path}")
    intv_dir = InterventionDirection.load(args.intv_path)
    print(f"  {len(intv_dir.components)} components loaded")

    llm = LLM(
        model=MODELS[args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_length + 2048,
        enforce_eager=True,
    )

    print("\nCollecting direction projections at last prompt token...")
    X, y = collect_prompt_projections(llm, questions, correctness, INSTRUCTION, intv_dir)

    if len(X) == 0:
        raise RuntimeError("No usable examples.")

    print("\nTraining prompt probe...")
    train_and_save_probe(X, y, args.probe_save_dir)


if __name__ == "__main__":
    main()
