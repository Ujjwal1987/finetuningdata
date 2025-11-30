import pandas as pd
import os
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, EarlyStoppingCallback

# Load the CSV file
main_file_path = os.path.join("Data", "training_data.csv")
df = pd.read_csv(main_file_path)

# Remove rows that contain at least one NaN value
df_cleaned = df.dropna()

# Create a copy to avoid SettingWithCopyWarning
df_cleaned = df_cleaned.copy()

# Clean specific columns by removing "[", "[SEP]", "]", and '""'
columns_to_clean = ['problem_verbatim', 'solution_verbatim', 'merged_claim_text']

for col in columns_to_clean:
    if col in df_cleaned.columns:
        df_cleaned.loc[:, col] = df_cleaned.loc[:, col].astype(str).str.replace('[', '', regex=False)
        df_cleaned.loc[:, col] = df_cleaned.loc[:, col].str.replace(']', '', regex=False)
        df_cleaned.loc[:, col] = df_cleaned.loc[:, col].str.replace('[SEP]', '', regex=False)
        df_cleaned.loc[:, col] = df_cleaned.loc[:, col].str.replace("'", '', regex=False)

json_list = []
for idx, row in df_cleaned.iterrows():
    json_obj = {
        "prompt": row['summary_text_y'] + " " + row['merged_claim_text'],
        "response": {  # <-- New 'response' header
            "problem": row['problem'],
            "problem_verbatim": row['problem_verbatim'],
            "solution": row['solution'],
            "solution_verbatim": row['solution_verbatim']
        }
    }
    json_list.append(json_obj)

# Save the JSON list to a file
with open('Data/output.json', 'w') as f:
    json.dump(json_list, f, indent=2)

print(f"Original rows: {len(df)}")
print(f"Rows after removing NaN: {len(df_cleaned)}")

# Optional: Save the cleaned dataframe back to a CSV file
Clean_file_path = os.path.join("Data", "cleaned_training_file.csv")
df_cleaned.to_csv(Clean_file_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Rows after removing NaN: {len(df_cleaned)}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'unsloth/gemma-3-270m-it',
    max_seq_length = 20480,
    dtype = None,
    load_in_4bit = False
)


with open("Data/output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)

def to_text(ex):
    resp = ex["response"]
    if not isinstance(resp, str):
        resp = json.dumps(resp, ensure_ascii=False)
    msgs = [
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": resp},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    }

dataset = ds.map(to_text, remove_columns=ds.column_names)

# Config From GitHub (without seed)
model = FastLanguageModel.get_peft_model(
    model,
    r = 256,  # rank of matrices (for LoRA)
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],  # which layers to inject LoRA into
    lora_alpha = 256 * 2,  # scaling factor, usually 2x rank
    lora_dropout = 0,  # no dropout, increase for regularizaiton
    bias = 'none',  # bias stays frozen, only learn the low-rank matrices
    use_gradient_checkpointing = 'unsloth',  # activate custom checkpointing scheme of Unsloth -> higher compute but less GPU memory when backpropagating
)


# 1. First, split your dataset into train/validation (do this before SFTTrainer)
ds_split = dataset.train_test_split(test_size=0.1) # 10% for validation (~130 records)
train_dataset = ds_split['train']
eval_dataset = ds_split['test']

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, # Added the validation set
    tokenizer = tokenizer,
    dataset_text_field = 'text',
    max_seq_length = 2048,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = -1, # REMOVED fixed steps to use epochs
        logging_steps = 10,
        output_dir = "outputs",
        optim = "adamw_8bit",
        num_train_epochs = 5, # Set a safe maximum like 5 epochs
        eval_strategy="epoch", # Corrected from evaluation_strategy
        save_strategy="epoch", # ADDED: Ensure save strategy matches eval strategy
        load_best_model_at_end=True, # Automatically uses the model that performed best on validation
        metric_for_best_model="eval_loss", # Which metric to monitor
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # Stop if loss doesn't improve for 2 epochs
)

trainer.train()