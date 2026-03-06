"""
train_probe.py — Headless probe training for ReflCtrl uncertainty-gated steering.

Loads baseline model outputs (from run_eval.py), feeds each (question, reasoning,
answer) triple back through the model to capture the </think> token's last-layer
embedding, then trains a logistic regression probe (correct vs. incorrect).

Saved outputs (compatible with ProbeMonitoringManager in hook_utils.py):
    <probe_save_dir>/clf_weights.pt   — numpy array, shape [hidden_dim]
    <probe_save_dir>/clf_bias.pt      — numpy array, shape [1]

Usage:
    python train_probe.py \\
        --model deepseek-r1-llama-8b \\
        --dataset gsm8k \\
        --results_path data/gsm8k/.../results_samples1.json \\
        --probe_save_dir probe_data/deepseek-r1-llama-8b/gsm8k/last_token_embedding
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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import MODELS

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train uncertainty probe from baseline outputs.")
    p.add_argument("--model", type=str, default="deepseek-r1-llama-8b")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--results_path", type=str, required=True,
                   help="Path to results_samples1.json from run_eval.py baseline run.")
    p.add_argument("--probe_save_dir", type=str,
                   default="probe_data/deepseek-r1-llama-8b/gsm8k/last_token_embedding")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)
    return p.parse_args()


def collect_think_end_embeddings(
    llm: LLM,
    questions: list[str],
    sample_results: dict,
    instruction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each question, feed the full (prompt + reasoning + </think> + answer) back
    through the model and capture the last-layer hidden state at the </think> token.

    Returns:
        X: float32 numpy array of shape [n_examples, hidden_dim]
        y: int numpy array of shape [n_examples], 1 = correct
    """
    tokenizer = llm.get_tokenizer()
    think_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # Store embeddings from the last transformer layer
    embeddings: list[torch.Tensor] = []
    handle_ref: list = []

    def register_hook(model):
        last_layer = model.model.layers[-1]

        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: [batch, seq_len, hidden_dim]
            embeddings.append(hidden[:, -1, :].float().cpu())

        handle_ref.append(last_layer.register_forward_hook(hook))

    llm.apply_model(register_hook)

    # SamplingParams: max_tokens=1 just to trigger the forward pass
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    response_texts = sample_results["response_texts"]
    think_texts    = sample_results["think_texts"]
    correctness    = sample_results["correctness"]

    X_list, y_list = [], []
    skipped = 0

    for i, (question, reasoning, answer, correct) in enumerate(
        zip(questions, think_texts, response_texts, correctness)
    ):
        # Build full text up to and including </think>
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + reasoning + "\n</think>\n" + answer

        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Find the </think> token position
        try:
            think_end_pos = token_ids.index(think_end_token_id)
        except ValueError:
            # </think> not present — skip this example
            skipped += 1
            continue

        # Feed only up to and including </think> to capture embedding there
        prefix_ids = token_ids[: think_end_pos + 1]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

        embeddings.clear()
        llm.generate(prefix_text, sp)

        if not embeddings:
            skipped += 1
            continue

        emb = embeddings[0].squeeze(0).numpy()  # [hidden_dim]
        X_list.append(emb)
        y_list.append(int(correct))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] collected {len(X_list)} examples, "
                  f"skipped {skipped}")

    # Remove hook
    if handle_ref:
        handle_ref[0].remove()

    print(f"Total: {len(X_list)} usable examples, {skipped} skipped (no </think>).")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_and_save_probe(X: np.ndarray, y: np.ndarray, save_dir: str) -> None:
    """Train logistic regression and save weights in ProbeMonitor-compatible format."""
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need both positive and negative examples. Got: {np.bincount(y)}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)

    val_scores = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_scores)
    val_acc = clf.score(X_val, y_val)
    print(f"Probe val accuracy: {val_acc:.4f}  |  AUROC: {auroc:.4f}")

    os.makedirs(save_dir, exist_ok=True)

    # Save in the format expected by ProbeMonitor in hook_utils.py:
    #   weights: [hidden_dim], bias: [1]
    weights = clf.coef_[0].astype(np.float32)   # shape [hidden_dim]
    bias    = clf.intercept_.astype(np.float32)  # shape [1]

    torch.save(weights, os.path.join(save_dir, "clf_weights.pt"))
    torch.save(bias,    os.path.join(save_dir, "clf_bias.pt"))

    # Also save metadata
    meta = {
        "val_accuracy": float(val_acc),
        "auroc": float(auroc),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }
    with open(os.path.join(save_dir, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Probe saved to {save_dir}/")
    print(f"  clf_weights.pt  shape: {weights.shape}")
    print(f"  clf_bias.pt     value: {bias}")


def main():
    args = parse_args()

    # Load previously generated baseline results
    print(f"Loading results from {args.results_path}")
    with open(args.results_path) as f:
        saved = json.load(f)

    questions = saved.get("questions")
    sample_results_list = saved.get("sample_results", [])

    if not sample_results_list:
        raise ValueError("No sample_results found in results file.")

    # We only use sample 0 (n_samples=1 for baseline)
    sample_results = sample_results_list[0]

    if questions is None:
        # Fallback: re-load questions from dataset
        from utils import extract_questions
        questions = extract_questions(args.dataset)
        n = len(sample_results["response_texts"])
        questions = questions[:n]

    print(f"Loaded {len(questions)} questions.")
    print(f"Correct: {sum(sample_results['correctness'])} / {len(sample_results['correctness'])}")

    # Initialise vLLM
    llm = LLM(
        model=MODELS[args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_length + 2048,
        enforce_eager=True,
    )

    # Collect embeddings
    print("\nCollecting </think> token embeddings...")
    X, y = collect_think_end_embeddings(llm, questions, sample_results, INSTRUCTION)

    if len(X) == 0:
        raise RuntimeError("No usable examples — check that baseline responses contain </think>.")

    # Train and save
    print("\nTraining probe...")
    train_and_save_probe(X, y, args.probe_save_dir)


if __name__ == "__main__":
    main()
