"""
train_probe.py — Headless probe training for ReflCtrl uncertainty-gated steering.

Matches the probe described in Section 3.5 of the ReflCtrl paper:

    Feature vector: [cos(d_l_attn, z_l_attn), cos(d_l_mlp, z_l_mlp)] for all layers l

where z_l is the activation at the </think> token position, and d_l is the
reflection direction vector for that layer. This gives a ~64-dim feature vector
(32 layers × 2) that the paper reports achieves AUROC 0.948 on GSM8k for
DeepSeek-R1-Llama-8B, compared to 0.736 for a raw last-layer embedding.

The probe is trained to predict whether the model's answer is correct (label=1)
or wrong (label=0).

Saved outputs (compatible with ProbeMonitoringManager in hook_utils.py):
    <probe_save_dir>/clf_weights.pt   — numpy array, shape [n_features]
    <probe_save_dir>/clf_bias.pt      — numpy array, shape [1]

Usage:
    python train_probe.py \\
        --model deepseek-r1-llama-8b \\
        --dataset gsm8k \\
        --results_path data/gsm8k/baseline/deepseek-r1-llama-8b/results_samples1.json \\
        --intv_path intervention_direction/deepseek-r1-llama-8b/gsm8k-train/reflect_dir.pt \\
        --probe_save_dir probe_data/deepseek-r1-llama-8b/gsm8k/reflect_dir
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
    p = argparse.ArgumentParser(description="Train direction-projection uncertainty probe.")
    p.add_argument("--model", type=str, default="deepseek-r1-llama-8b")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--results_path", type=str, required=True,
                   help="Path to results_samples1.json from run_eval.py baseline run.")
    p.add_argument("--intv_path", type=str, required=True,
                   help="Path to reflect_dir.pt (InterventionDirection with direction vectors).")
    p.add_argument("--probe_save_dir", type=str,
                   default="probe_data/deepseek-r1-llama-8b/gsm8k/reflect_dir",
                   help="Directory to save clf_weights.pt and clf_bias.pt.")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)
    return p.parse_args()


def collect_direction_projections(
    llm: LLM,
    questions: list[str],
    sample_results: dict,
    instruction: str,
    intv_dir: InterventionDirection,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each question, feed (prompt + reasoning + </think> + answer) through the
    model and capture — for each layer — the dot product of the activation at the
    </think> token with the normalised reflection direction vector.

    This matches the paper's feature:
        p_l = cos(d_l, z_l) = z_l · (d_l / |d_l|)

    Returns:
        X: float32 array of shape [n_examples, n_components]
           where n_components = 2 * n_layers (attn + mlp per layer)
        y: int array of shape [n_examples], 1 = correct
    """
    tokenizer = llm.get_tokenizer()
    think_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    # Build normalised direction dict: component_name -> [hidden_dim] tensor
    component_names = sorted(intv_dir.components.keys())
    direction_vectors = {
        comp: (
            intv_dir.components[comp].mean_diff
            / intv_dir.components[comp].mean_diff.norm()
        ).float()
        for comp in component_names
    }

    # proj_buffer[component_name] = scalar projection at the last processed token
    proj_buffer: dict[str, float] = {}
    handle_refs: list = []

    def make_hook(comp_name: str, direction: torch.Tensor):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if hidden.dim() == 3:
                last = hidden[:, -1, :].float()   # [batch, hidden_dim]
            else:
                last = hidden[-1, :].float().unsqueeze(0)  # [1, hidden_dim]
            # Scalar projection onto the unit direction vector
            proj = (last @ direction.to(last.device)).squeeze()
            proj_buffer[comp_name] = proj.item()
        return hook

    def register_all_hooks(model):
        for comp_name, direction in direction_vectors.items():
            module = eval(f"model.{comp_name}")
            handle = module.register_forward_hook(make_hook(comp_name, direction))
            handle_refs.append(handle)

    llm.apply_model(register_all_hooks)

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    response_texts = sample_results["response_texts"]
    think_texts    = sample_results["think_texts"]
    correctness    = sample_results["correctness"]

    X_list, y_list = [], []
    skipped = 0

    for i, (question, reasoning, answer, correct) in enumerate(
        zip(questions, think_texts, response_texts, correctness)
    ):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Reconstruct the full trace and find </think> position
        full_text = prompt + reasoning + "\n</think>\n" + answer
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        try:
            think_end_pos = token_ids.index(think_end_token_id)
        except ValueError:
            skipped += 1
            continue

        # Feed only the prefix up to and including </think>.
        # The hook captures activations at the last position = </think> token.
        prefix_ids = token_ids[: think_end_pos + 1]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

        proj_buffer.clear()
        llm.generate(prefix_text, sp)

        if len(proj_buffer) < len(component_names):
            skipped += 1
            continue

        # Feature vector: projections in sorted component order
        feature = [proj_buffer[c] for c in component_names]
        X_list.append(feature)
        y_list.append(int(correct))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] {len(X_list)} examples collected, "
                  f"{skipped} skipped (no </think>)")

    # Remove hooks
    for h in handle_refs:
        h.remove()

    print(f"\nTotal: {len(X_list)} usable examples, {skipped} skipped.")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def train_and_save_probe(X: np.ndarray, y: np.ndarray, save_dir: str) -> None:
    """Train logistic regression and save weights in ProbeMonitor-compatible format."""
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need both correct and incorrect examples. Got: {np.bincount(y)}")

    print(f"Feature vector size: {X.shape[1]} ({X.shape[1]//2} layers × 2 components)")
    print(f"Class balance: {y.sum()} correct, {(1-y).sum()} incorrect out of {len(y)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)

    val_scores = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_scores)
    val_acc = clf.score(X_val, y_val)
    print(f"Probe val accuracy: {val_acc:.4f}  |  AUROC: {auroc:.4f}")
    print(f"(Paper reports AUROC 0.948 for DeepSeek-R1-Llama-8B with this feature type)")

    os.makedirs(save_dir, exist_ok=True)

    # Save in format expected by ProbeMonitor in hook_utils.py:
    #   weights: [n_features], bias: [1]
    weights = clf.coef_[0].astype(np.float32)
    bias    = clf.intercept_.astype(np.float32)

    torch.save(weights, os.path.join(save_dir, "clf_weights.pt"))
    torch.save(bias,    os.path.join(save_dir, "clf_bias.pt"))

    meta = {
        "val_accuracy": float(val_acc),
        "auroc": float(auroc),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "feature_dim": int(X.shape[1]),
        "feature_type": "direction_projection",
    }
    with open(os.path.join(save_dir, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nProbe saved to {save_dir}/")


def main():
    args = parse_args()

    # Load baseline results
    print(f"Loading results from {args.results_path}")
    with open(args.results_path) as f:
        saved = json.load(f)

    questions = saved.get("questions")
    sample_results_list = saved.get("sample_results", [])
    if not sample_results_list:
        raise ValueError("No sample_results found in results file.")

    sample_results = sample_results_list[0]  # first (and only) sample

    if questions is None:
        questions = extract_questions(args.dataset)
        n = len(sample_results["response_texts"])
        questions = questions[:n]

    print(f"Loaded {len(questions)} questions  |  "
          f"correct: {sum(sample_results['correctness'])}/{len(sample_results['correctness'])}")

    # Load direction vectors
    print(f"\nLoading direction vectors from {args.intv_path}")
    intv_dir = InterventionDirection.load(args.intv_path)
    print(f"  {len(intv_dir.components)} components loaded")

    # Initialise vLLM
    llm = LLM(
        model=MODELS[args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_length + 2048,
        enforce_eager=True,
    )

    # Collect features
    print("\nCollecting direction projections at </think> token...")
    X, y = collect_direction_projections(llm, questions, sample_results, INSTRUCTION, intv_dir)

    if len(X) == 0:
        raise RuntimeError(
            "No usable examples — check that baseline responses contain </think> tokens."
        )

    # Train and save
    print("\nTraining probe...")
    train_and_save_probe(X, y, args.probe_save_dir)


if __name__ == "__main__":
    main()
