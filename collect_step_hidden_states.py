"""
collect_step_hidden_states.py — Collect full hidden states at \\n\\n step boundaries.

For each question in the baseline results, feeds the full trace (prompt + reasoning)
through the model and saves the 4096-dim hidden state at every \\n\\n token position
for each layer/module. These are used to train per-layer uncertainty probes.

Each \\n\\n token from a question inherits that question's correctness label.

Output:
    <output_dir>/layer_hidden_states.pt — dict with:
        'hidden_states': {component_name: tensor [n_total_steps, hidden_dim]}
        'labels': tensor [n_total_steps] (1=correct, 0=incorrect)
        'question_ids': tensor [n_total_steps] (which question each step came from)

Usage:
    python collect_step_hidden_states.py \
        --model deepseek-r1-llama-8b \
        --results_path data/adaptive_results/baseline/results_samples1.json \
        --output_dir probe_data/deepseek-r1-llama-8b/gsm8k/step_hidden_states
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from hook_utils import MODEL_LAYER_MAP, ActivationCacher
from utils import MODELS, extract_questions

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect hidden states at step boundaries.")
    p.add_argument("--model", type=str, default="deepseek-r1-llama-8b")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--results_path", type=str, required=True,
                   help="Path to baseline results_samples1.json")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save hidden states")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)
    return p.parse_args()


def main():
    args = parse_args()

    # Load baseline results
    print(f"Loading results from {args.results_path}")
    with open(args.results_path) as f:
        saved = json.load(f)

    questions = saved.get("questions")
    sample_results = saved["sample_results"][0]
    correctness = sample_results["correctness"]
    think_texts = sample_results["think_texts"]

    if questions is None:
        questions = extract_questions(args.dataset)
        questions = questions[:len(correctness)]

    print(f"Loaded {len(questions)} questions  |  "
          f"correct: {sum(correctness)}/{len(correctness)}")

    # Init vLLM
    llm = LLM(
        model=MODELS[args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_length + 2048,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Find \n\n token IDs
    newline_token_ids = set(
        tid for tid in range(tokenizer.vocab_size)
        if "\n\n" in tokenizer.decode(tid)
    )
    print(f"Found {len(newline_token_ids)} \\n\\n token IDs")

    # Register activation hooks for all layers
    target_modules = MODEL_LAYER_MAP[args.model]
    cacher = ActivationCacher()
    llm.apply_model(lambda model: cacher.register_model(model, target_modules))
    print(f"Registered hooks on {len(target_modules)} modules")

    # Collect hidden states at \n\n positions
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Storage: {component: list of [hidden_dim] tensors}
    all_hidden_states = defaultdict(list)
    all_labels = []
    all_question_ids = []
    skipped = 0

    for i, (question, reasoning, correct) in enumerate(
        tqdm(zip(questions, think_texts, correctness), total=len(questions),
             desc="Collecting hidden states")
    ):
        # Build full token sequence: prompt + <think>\n + reasoning
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + INSTRUCTION}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + "<think>\n" + reasoning
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Find \n\n positions in the output (after prompt)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        offset = len(prompt_ids)

        # Get positions of \n\n tokens in the output portion
        step_positions = [
            j for j, tid in enumerate(token_ids[offset:])
            if tid in newline_token_ids
        ]

        if not step_positions:
            skipped += 1
            continue

        # Run forward pass on the full sequence
        cacher.clear_cache()
        token_prompt = TokensPrompt(prompt_token_ids=token_ids)
        llm.generate(token_prompt, sp)

        # Extract hidden states at each \n\n position
        cache = cacher.get_cache()
        valid = True
        for component in target_modules:
            if not cache[component]:
                valid = False
                break
            # cache[component][0] has shape [seq_len, hidden_dim]
            act = cache[component][0]
            if act.shape[0] < offset + max(step_positions) + 1:
                valid = False
                break

        if not valid:
            skipped += 1
            continue

        for pos in step_positions:
            for component in target_modules:
                act = cache[component][0]
                hidden = act[offset + pos].clone()
                all_hidden_states[component].append(hidden)
            all_labels.append(int(correct))
            all_question_ids.append(i)

        if (i + 1) % 50 == 0:
            n_steps = len(all_labels)
            print(f"  [{i+1}/{len(questions)}] {n_steps} step samples collected, "
                  f"{skipped} questions skipped")

    print(f"\nTotal: {len(all_labels)} step samples from "
          f"{len(set(all_question_ids))} questions, {skipped} skipped")
    print(f"Class balance: {sum(all_labels)} correct steps, "
          f"{len(all_labels) - sum(all_labels)} incorrect steps")

    # Stack into tensors
    hidden_states_dict = {}
    for component in target_modules:
        hidden_states_dict[component] = torch.stack(all_hidden_states[component])
        print(f"  {component}: {hidden_states_dict[component].shape}")

    labels = torch.tensor(all_labels, dtype=torch.long)
    question_ids = torch.tensor(all_question_ids, dtype=torch.long)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "layer_hidden_states.pt")
    torch.save({
        'hidden_states': hidden_states_dict,
        'labels': labels,
        'question_ids': question_ids,
    }, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
