#!/bin/bash

# Collect activations from gsm8k training set and generate intervention directions
# Configuration
MODEL="deepseek-r1-llama-8b"
DATASET="gsm8k"
INSTRUCTION="Please reason step by step, and put your final answer within \boxed{}."

# Step 1: Collect activations
echo "Collecting activations..."
python collect_activation.py \
    --model $MODEL \
    --dataset $DATASET \
    --instruction "$INSTRUCTION"

# Step 2: Extract intervention directions
echo "Extracting intervention directions..."
python extract_dir.py \
    --model $MODEL \
    --dataset $DATASET \
    --instruction "$INSTRUCTION"

echo "Done!"