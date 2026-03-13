"""
majority_vote.py — Compute majority-vote accuracy from saved results files.

Works with both:
  - n=10 results files (sample_results has 10 entries)
  - aggregated n=1 runs (from aggregate_runs.py)

Usage:
    python majority_vote.py results_samples10.json
    python majority_vote.py file1.json file2.json file3.json
"""

import json
import sys
from collections import Counter


def majority_vote_from_file(path):
    """Compute majority vote from a single results file with multiple samples."""
    with open(path) as f:
        data = json.load(f)

    sample_results = data["sample_results"]
    n_samples = len(sample_results)
    n_questions = len(sample_results[0]["correctness"])

    # Per-question majority vote
    majority_correct = 0
    for q in range(n_questions):
        votes = sum(sample_results[s]["correctness"][q] for s in range(n_samples))
        if votes > n_samples / 2:
            majority_correct += 1

    accuracy = majority_correct / n_questions

    # Mean accuracy (for comparison)
    mean_acc = sum(s["accuracy"] for s in sample_results) / n_samples

    # Average thinking length
    avg_tokens = sum(s["avg_thinking_length"] for s in sample_results) / n_samples

    return {
        "majority_vote_accuracy": accuracy,
        "mean_accuracy": mean_acc,
        "avg_thinking_length": avg_tokens,
        "n_samples": n_samples,
        "n_questions": n_questions,
    }


def majority_vote_from_multiple_files(paths):
    """Compute majority vote from multiple n=1 results files."""
    all_correctness = []
    all_thinking = []

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        sample = data["sample_results"][0]
        all_correctness.append(sample["correctness"])
        all_thinking.append(sample["avg_thinking_length"])

    n_runs = len(all_correctness)
    n_questions = len(all_correctness[0])

    majority_correct = 0
    for q in range(n_questions):
        votes = sum(all_correctness[r][q] for r in range(n_runs))
        if votes > n_runs / 2:
            majority_correct += 1

    accuracy = majority_correct / n_questions
    mean_acc = sum(
        sum(c) / len(c) for c in all_correctness
    ) / n_runs
    avg_tokens = sum(all_thinking) / n_runs

    return {
        "majority_vote_accuracy": accuracy,
        "mean_accuracy": mean_acc,
        "avg_thinking_length": avg_tokens,
        "n_samples": n_runs,
        "n_questions": n_questions,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python majority_vote.py <results_file(s)>")
        sys.exit(1)

    paths = sys.argv[1:]

    if len(paths) == 1:
        # Single file with multiple samples
        result = majority_vote_from_file(paths[0])
    else:
        # Multiple n=1 files
        result = majority_vote_from_multiple_files(paths)

    print(f"  n_samples:             {result['n_samples']}")
    print(f"  n_questions:           {result['n_questions']}")
    print(f"  mean_accuracy:         {result['mean_accuracy']:.4f}")
    print(f"  majority_vote_accuracy:{result['majority_vote_accuracy']:.4f}")
    print(f"  avg_thinking_length:   {result['avg_thinking_length']:.1f} tokens")
