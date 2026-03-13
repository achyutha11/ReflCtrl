"""
aggregate_runs.py — Aggregate multiple n=1 runs into majority-vote results.

Takes multiple results_samples1.json files (from repeated n=1 runs) and
combines them as if they were n samples from a single run, computing
majority-vote accuracy.

Usage:
    python aggregate_runs.py \
        --run_dirs run_1 run_2 run_3 ... run_10 \
        --dataset gsm8k \
        --output_path aggregated_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from utils import extract_questions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multiple n=1 runs.")
    p.add_argument("--run_dirs", type=str, nargs="+", required=True,
                   help="Directories containing results_samples1.json")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--output_path", type=str, default=None,
                   help="Path to save aggregated results (default: print only)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load all runs
    all_runs = []
    for d in args.run_dirs:
        path = os.path.join(d, "results_samples1.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            all_runs.append(json.load(f))

    n_runs = len(all_runs)
    if n_runs == 0:
        print("No valid runs found!")
        return

    print(f"Loaded {n_runs} runs")

    # Each run has sample_results: list of 1 sample dict
    # sample_results[0] has 'correctness', 'think_texts', etc.
    n_questions = len(all_runs[0]["sample_results"][0]["correctness"])
    print(f"{n_questions} questions per run")

    # Collect per-question correctness across runs
    all_correctness = []  # [n_runs][n_questions]
    all_thinking_lengths = []  # [n_runs][n_questions]

    for run in all_runs:
        sample = run["sample_results"][0]
        all_correctness.append(sample["correctness"])
        all_thinking_lengths.append(sample["think_lengths"])

    # Majority vote: question is correct if majority of runs got it right
    majority_correct = 0
    per_question = []
    for q in range(n_questions):
        votes = [all_correctness[r][q] for r in range(n_runs)]
        n_correct = sum(votes)
        is_correct = n_correct > n_runs / 2  # strict majority
        if is_correct:
            majority_correct += 1
        per_question.append({
            "n_correct": n_correct,
            "n_runs": n_runs,
            "majority_correct": is_correct,
        })

    # Average thinking length across all runs
    total_thinking = 0
    total_samples = 0
    for r in range(n_runs):
        for q in range(n_questions):
            total_thinking += all_thinking_lengths[r][q]
            total_samples += 1
    avg_thinking = total_thinking / total_samples if total_samples > 0 else 0

    # Per-run accuracy
    print(f"\nPer-run accuracy:")
    for i, run in enumerate(all_runs):
        acc = run["aggregate"]["accuracy"]
        tokens = run["aggregate"]["avg_thinking_length"]
        print(f"  Run {i+1}: {acc:.4f} ({tokens:.1f} tokens)")

    accuracy = majority_correct / n_questions
    print(f"\nMajority vote ({n_runs} runs):")
    print(f"  accuracy:            {accuracy:.4f}")
    print(f"  avg_thinking_length: {avg_thinking:.1f} tokens")
    print(f"  n_questions:         {n_questions}")

    if args.output_path:
        results = {
            "aggregate": {
                "accuracy": accuracy,
                "avg_thinking_length": avg_thinking,
                "n_runs": n_runs,
                "n_questions": n_questions,
            },
            "per_run": [
                {"accuracy": r["aggregate"]["accuracy"],
                 "avg_thinking_length": r["aggregate"]["avg_thinking_length"]}
                for r in all_runs
            ],
            "per_question": per_question,
        }
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
