import torch
# print("Torch:", torch.__version__)
# print("HIP:", torch.version.hip)
# print("CUDA available:", torch.cuda.is_available())
# print("Num devices:", torch.cuda.device_count())

print(f'Torch Version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')

from trl import SFTTrainer
help(SFTTrainer.__init__)

import os
import json
import pandas as pd
import numpy as np  # Added for token length analysis
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. AMD ROCm stability and memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Keeping this for VRAM management/fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

###############################################################
# LOAD, CLEAN, and CONVERT DATA
###############################################################
try:
    df = pd.read_csv("Data/training_data.csv")
except FileNotFoundError:
    print("Error: training_data.csv not found in 'Data/' directory. Please check the path.")
    exit()

df_cleaned = df.dropna().copy()

clean_cols = ["problem_verbatim", "solution_verbatim", "merged_claim_text"]
for col in clean_cols:
    df_cleaned[col] = (
        df_cleaned[col]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("[SEP]", "", regex=False)
        .str.replace("'", "", regex=False)
    )


def convert_example(row):
    # Ensure all data in the response is correctly encapsulated in a JSON string
    response_data = {
        "problem": row["problem_verbatim"],
        "problem_verbatim": row["problem_verbatim"],
        "solution": row["solution_verbatim"],
        "solution_verbatim": row["solution_verbatim"],
    }
    response = json.dumps(response_data, ensure_ascii=False)
    # Full instruction format: <s>[INST] Prompt [/INST] Response</s>
    text = f"<s>[INST]{row['summary_text_y']} {row['merged_claim_text']}[/INST]{response}</s>"
    return {"text": text}


dataset = Dataset.from_pandas(df_cleaned)
dataset = dataset.map(convert_example, remove_columns=dataset.column_names)
ds_split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = ds_split["train"]
eval_ds = ds_split["test"]

print(f"Train dataset size: {len(train_ds)}")
print(f"Eval dataset size: {len(eval_ds)}")

###############################################################
# TOKEN LENGTH ANALYSIS
###############################################################

model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Initialize model_max_length to the full context window for analysis
tokenizer.model_max_length = 6144


def calculate_token_statistics(dataset, tokenizer):
    """Tokenizes all samples in a Hugging Face Dataset and returns length percentiles."""
    print("Starting token length analysis on training data...")
    token_lengths = []

    # Extract the 'text' column which contains the full fine-tuning sample string
    texts = dataset['text']

    # Use the tokenizer to get the length of the input_ids for each text
    for text in texts:
        tokens = tokenizer(text, truncation=False)
        token_lengths.append(len(tokens["input_ids"]))

    token_lengths = np.array(token_lengths)

    if len(token_lengths) == 0:
        print("Warning: No samples found for token analysis.")
        return None

    # Calculate key percentiles
    p90 = np.percentile(token_lengths, 90)
    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)

    print("\n--- Token Length Statistics (Training Dataset) ---")
    print(f"Total Samples Analyzed: {len(token_lengths)}")
    print(f"Min Length: {np.min(token_lengths)} tokens")
    print(f"Max Length: {np.max(token_lengths)} tokens")
    print(f"Mean Length: {np.mean(token_lengths):.0f} tokens")
    print("-" * 30)
    print(f"90th Percentile (P90): {p90:.0f} tokens (Setting max_length here would truncate 10% of data)")
    print(f"95th Percentile (P95): {p95:.0f} tokens (Setting max_length here would truncate 5% of data)")
    print(f"99th Percentile (P99): {p99:.0f} tokens (Setting max_length here would truncate 1% of data)")
    print("-" * 30)
    print(
        f"Recommendation: To retain 95% of your data while optimizing VRAM, set tokenizer.model_max_length to approximately {p95:.0f} tokens. Use 6144 if VRAM allows.")

    return p95


# Run the analysis on the training dataset
suggested_max_length_p95 = calculate_token_statistics(train_ds, tokenizer)

# --- Apply the chosen max_length ---
# If you decide to change the sequence length based on the analysis (e.g., to 2048),
# uncomment and modify the line below. We keep 6144 for now as per the original code.
# tokenizer.model_max_length = 6144
print(f"\nContinuing fine-tuning with tokenizer.model_max_length = {tokenizer.model_max_length}...")