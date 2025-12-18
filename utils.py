import json
import math
import os
import re
import numpy as np
from datasets import load_dataset

from math_grader import math_equal, strip_string
THINK_START_ID, THINK_END_ID = 128798, 128799

DATASET_MAP = {
    "gsm8k": {"args": ("openai/gsm8k", "main"), "question_key": "question", "answer_key": "answer", "split": "test"},
    "gsm8k-train": {"args": ("openai/gsm8k", "main"), "question_key": "question", "answer_key": "answer", "split": "train"},
    "MATH-500": {"args": ("HuggingFaceH4/MATH-500",), "question_key": "problem", "answer_key": "answer", "split": "test"},
    "AIME24": {"args": ("Maxwell-Jia/AIME_2024",), "question_key": "Problem", "answer_key": "Answer", "split": "train"},
    "MMLU": {"args": ("cais/mmlu",), "question_key": "question", "answer_key": "answer", "split": "test"},
    "openr1-math": {"args": ("open-r1/OpenR1-Math-220k", "default"), "question_key": "problem", "answer_key": "answer", "split": "train[:10000]"},
    "gpqa": {"args": ("fingertap/GPQA-Diamond", ), "question_key": "question", "answer_key": "answer", "split": "test"}
}
REFLECT_WORDS = ["wait", "let me check", "double-check", "alternatively"]
END_WORDS = ["final answer"]
MODELS = {
    "deepseek-r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "QwQ-32b": "Qwen/QwQ-32B",
    "llama-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
}
def load_results(model, dataset, instruction=""):
    data_path = f"data/{dataset}/short_thinking_attn_orthogonal_ablation/deepseek-r1-{model}/instruction_{instruction}"
    interv_path = f"data/{dataset}/short_thinking_attn_orthogonal_ablation/intervened-{model}/instruction_{instruction}"
    result_name = "results_samples1.json"
    base_path = os.path.join(data_path, result_name)
    interv_path = os.path.join(interv_path, result_name)

    with open(base_path, "r") as f:
        base_data = json.load(f)
    with open(interv_path, "r") as f:
        interv_data = json.load(f)

    return base_data, interv_data

def extract_answer_mmlu(text):
    ans = extract_boxed(text)
    if ans:
        return ans[-1][1]
    patterns = [r"Answer: (\w)", r"Answer: \*\*(\w)\*\*", r"\*\*Answer:\*\* (\w)", r" ([ABCD])\."]
    for pattern in patterns:
        pattern = re.compile(pattern)
        match = pattern.search(text)
        if match:
            return match.group(1)
    else: return None


def construct_mmlu_prompt(question, choices, subject):
    markers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{markers[i]}. {choice}\n"
    return prompt

def extract_questions(dataset):
    if dataset.startswith("MMLU"):
        dataset, subject = "MMLU", dataset[5:]
        dataset = load_dataset(*DATASET_MAP[dataset]["args"], subject, split=DATASET_MAP[dataset]["split"])
        questions = dataset["question"]
        choices_list = dataset["choices"]
        questions = [construct_mmlu_prompt(question, choices, subject) for question, choices in zip(questions, choices_list)]
    else:
        question_key = DATASET_MAP[dataset]["question_key"]
        dataset = load_dataset(*DATASET_MAP[dataset]["args"], split=DATASET_MAP[dataset]["split"])
        questions = list(dataset[question_key])
    return questions
    

def extract_answer_math(text):
    if text is None:
        return None
    # Step 1: Remove everything that is not a number, letter, ".", or "-"
    # text = re.sub(r'[^0-9a-zA-Z{}\\.\-]', '', text)
    # Try extracting from 'boxed' first
    boxed_matches = extract_boxed(text)
    if boxed_matches:
        extracted_answer = boxed_matches[-1][1:-1]
        return strip_string(extracted_answer)

    # Fallback: extract any numbers
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if not numbers:
        return None

    try:
        extracted_number = float(numbers[-1])
        # Guard against infinity
        if math.isinf(extracted_number):
            return None
        
        return numbers[-1]
    except (ValueError, OverflowError):
        return None


def analyze_math_results(responses, dataset_name):
    """
    Analyze results for multiple samples per question.
    
    Args:
        responses: List of lists, where each inner list contains responses for one sample
        dataset_name: Name of the dataset
    """
    if dataset_name.startswith("MMLU"):
        dataset_name, subject = "MMLU", dataset_name[5:]
        dataset = load_dataset(*DATASET_MAP[dataset_name]["args"], subject, split=DATASET_MAP[dataset_name]["split"])
    else:
        dataset = load_dataset(*DATASET_MAP[dataset_name]["args"], split=DATASET_MAP[dataset_name]["split"])
    # Get ground truth answers
    answer_key = DATASET_MAP[dataset_name]["answer_key"]
    if dataset_name == "gsm8k" or dataset_name == "gsm8k-train":
        answers = [str(ex[answer_key]).split('####')[-1].strip() for ex in dataset]
    else:
        answers = dataset[answer_key]
    answers = [strip_string(str(true)) for true in answers]
    
    # Process each sample
    all_stats = []
    choices= ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for sample_responses in responses:
        response_texts = [resp['content'] for resp in sample_responses]
        thinking_texts = [resp['reasoning'] for resp in sample_responses]
        thinking_lengths = [resp['thinking_length'] for resp in sample_responses]
        
        # Extract predictions for this sample
        if dataset_name.startswith("MMLU") or dataset_name == "gpqa":
            predicted = [extract_answer_mmlu(resp) for resp in response_texts]
        else:
            predicted = [extract_answer_math(resp) for resp in response_texts]
        
        # Compare predictions to ground truth
        correctness = []
        for pred, true in zip(predicted, answers):
            if pred is None:
                correctness.append(False)
            else:
                if dataset_name.startswith("MMLU"):
                    correctness.append(pred == choices[int(true)])
                elif dataset_name == "gpqa":
                    correctness.append(pred == true)
                else:
                    try:
                        correctness.append(math_equal(pred, true))
                    except:
                        correctness.append(False)
        
        sample_stats = {
            'accuracy': np.mean(np.array(correctness)),
            'avg_thinking_length': np.mean(thinking_lengths),
            'think_lengths': thinking_lengths,
            'think_texts': thinking_texts,
            'response_texts': response_texts,
            'correctness': correctness,
            'predicted': predicted,
        }
        all_stats.append(sample_stats)
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'accuracy': np.mean([stats['accuracy'] for stats in all_stats]),
        'avg_thinking_length': np.mean([stats['avg_thinking_length'] for stats in all_stats]),
    }
    
    analyzed_results = {
        "sample_results": all_stats,
        "answers": answers,
    }
    
    return aggregate_stats, analyzed_results


def extract_boxed(text):
    pattern = re.compile(r'boxed\{')
    matches = []
    stack = []
    
    i = 0
    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break
        
        start = match.end() - 1  # Position at the first `{`
        stack.append(start)
        i = start + 1
        count = 1  # To track `{}` pairs
        
        while i < len(text) and stack:
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:  # Found a matching closing `}`
                    start = stack.pop()
                    matches.append(text[start:i+1])
                    break
            i += 1
    
    return matches


def remove_text(text):
    return re.sub(r'\\text{.*?}', '', text)


def get_think_length(output_ids, think_start_id=THINK_START_ID,
                     think_end_id=THINK_END_ID, max_length=8192):
    think_starts = [i for i, token in enumerate(output_ids) if token == think_start_id]
    think_ends = [i for i, token in enumerate(output_ids) if token == think_end_id]
    
    if think_starts and think_ends:
        return think_ends[0] - think_starts[0] + 1, True
    elif think_starts and not think_ends:
        return max_length, False
    elif not think_starts and think_ends:
        return think_ends[0] + 1, False
    else:
        return len(output_ids), False


def get_save_dir(dataset: str, model: str, instruction: str, with_intervention: float = 0, 
               intervention_direction: str = "reflect", intervention_layers: str = None, 
               step_begin_only: bool = False, intervention_type: str = "additive", nowait: bool = False,
               intv_path: str = None) -> str:
    """
    Get the save directory path based on the given parameters.
    
    Args:
        dataset: Name of the dataset
        model: Name of the model
        instruction: Instruction string
        with_intervention: Intervention strength (default: 0)
        intervention_layers: Layer range for intervention (default: None)
        step_begin_only: Whether to only intervene at step beginning (default: False)
        
    Returns:
        str: Path to the save directory
    """
    save_dir = f"data/{dataset}/short_thinking_attn_orthogonal_ablation/{model}/instruction_{instruction}"
    if with_intervention != 0:
        save_dir += f"/with_intervention_{with_intervention}"
    if intervention_layers is not None:
        save_dir += f"/layers_{intervention_layers}"
    if step_begin_only:
        save_dir += "/step_begin_only"
    if intervention_direction is not None:
        save_dir += f"/{intervention_direction}_dir"
    if intervention_type != "additive":
        save_dir += f"/{intervention_type}_intervention"
    if nowait:
        save_dir += "/nowait"
    if intv_path is not None:
        save_dir += f"/{intv_path.split('/')[-1].split('.')[0]}"
    return save_dir