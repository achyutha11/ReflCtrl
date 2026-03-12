"""
train_layer_probes.py — Train per-layer uncertainty probes on full hidden states.

Takes the hidden states collected by collect_step_hidden_states.py and trains
64 separate logistic regression classifiers (one per layer/module), each using
the full 4096-dim hidden state to predict correctness.

Outputs per-layer AUROC (useful for diagnostics — which layers encode uncertainty?)
and saves weights for all probes.

Output:
    <output_dir>/layer_probes/
        <component_name>/clf_weights.pt, clf_bias.pt
        probe_summary.json  — per-layer AUROC and accuracy

Usage:
    python train_layer_probes.py \
        --hidden_states_path probe_data/deepseek-r1-llama-8b/gsm8k/step_hidden_states/layer_hidden_states.pt \
        --output_dir probe_data/deepseek-r1-llama-8b/gsm8k/layer_probes
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-layer uncertainty probes.")
    p.add_argument("--hidden_states_path", type=str, required=True,
                   help="Path to layer_hidden_states.pt from collect_step_hidden_states.py")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save per-layer probe weights")
    p.add_argument("--C", type=float, default=1.0,
                   help="Logistic regression regularization strength (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading hidden states from {args.hidden_states_path}")
    data = torch.load(args.hidden_states_path, weights_only=False)

    hidden_states = data['hidden_states']  # dict: component -> [n_steps, hidden_dim]
    labels = data['labels'].numpy()        # [n_steps]
    question_ids = data['question_ids'].numpy()  # [n_steps]

    n_steps = len(labels)
    n_questions = len(np.unique(question_ids))
    components = sorted(hidden_states.keys())

    print(f"  {n_steps} step samples from {n_questions} questions")
    print(f"  {len(components)} components")
    print(f"  Class balance: {labels.sum()} correct, {n_steps - labels.sum()} incorrect")

    # Split by question (not by step) to avoid data leakage
    # All steps from the same question must be in the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(np.arange(n_steps), labels, groups=question_ids))

    y_train, y_val = labels[train_idx], labels[val_idx]
    print(f"  Train: {len(train_idx)} steps, Val: {len(val_idx)} steps")
    print(f"  Train balance: {y_train.sum()} correct, {len(y_train) - y_train.sum()} incorrect")
    print(f"  Val balance: {y_val.sum()} correct, {len(y_val) - y_val.sum()} incorrect")

    # Train one probe per component
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTraining {len(components)} per-layer probes (C={args.C})...\n")

    for component in components:
        X = hidden_states[component].float().numpy()
        X_train, X_val = X[train_idx], X[val_idx]

        clf = LogisticRegression(max_iter=1000, C=args.C, solver='lbfgs')
        clf.fit(X_train, y_train)

        val_proba = clf.predict_proba(X_val)[:, 1]
        try:
            auroc = roc_auc_score(y_val, val_proba)
        except ValueError:
            auroc = 0.5  # only one class in val
        val_acc = clf.score(X_val, y_val)

        # Save probe weights
        comp_dir = os.path.join(args.output_dir, component.replace(".", "_").replace("[", "_").replace("]", "_"))
        os.makedirs(comp_dir, exist_ok=True)

        weights = clf.coef_[0].astype(np.float32)
        bias = clf.intercept_.astype(np.float32)
        torch.save(weights, os.path.join(comp_dir, "clf_weights.pt"))
        torch.save(bias, os.path.join(comp_dir, "clf_bias.pt"))

        results[component] = {
            'auroc': float(auroc),
            'val_accuracy': float(val_acc),
        }

        # Short name for display
        short = component.split(".")[-1]
        layer_num = component.split("[")[1].split("]")[0] if "[" in component else "?"
        print(f"  Layer {layer_num:>2s} {short:<10s}  AUROC: {auroc:.4f}  acc: {val_acc:.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "probe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print overall stats
    aurocs = [r['auroc'] for r in results.values()]
    print(f"\nAUROC summary:")
    print(f"  Mean: {np.mean(aurocs):.4f}")
    print(f"  Max:  {np.max(aurocs):.4f} ({max(results, key=lambda k: results[k]['auroc'])})")
    print(f"  Min:  {np.min(aurocs):.4f}")
    print(f"  >0.7: {sum(1 for a in aurocs if a > 0.7)} / {len(aurocs)} components")
    print(f"  >0.8: {sum(1 for a in aurocs if a > 0.8)} / {len(aurocs)} components")

    print(f"\nSaved {len(components)} probes to {args.output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
