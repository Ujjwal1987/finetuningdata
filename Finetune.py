import os
import random
import pandas as pd
import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,  # <--- NEW: Required for QLoRA
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training  # <--- NEW: Required for QLoRA stability
)
from trl import SFTTrainer

# 1. AMD ROCm stability and memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

###############################################################
# DATA LOADING & PROCESSING (Your Custom Logic Preserved)
###############################################################
df = pd.read_csv("Data/training_data.csv")
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
    # Problem-focused variations
    # problem_variations = [
    #     f"Identify the technical problem from {row['summary_text_y']}",
    #     f"What is the main problem addressed by this patent? {row['summary_text_y']}",
    #     f"Problem addressed by this patent claim: {row['summary_text_y']}",
    #     f"Main technical challenge solved by: {row['summary_text_y']} ",
    #     f"What technical problem does this patent solve? {row['summary_text_y']}",
    #     f"Identify problem {row['summary_text_y']} ",
    #     f"problem {row['summary_text_y']}"
    # ]

    # Solution-focused variations
    solution_variations = [
        f"Extract the technical solution from: {row['summary_text_y']} {row['merged_claim_text']}",
        f"What is the proposed solution in this patent? {row['summary_text_y']} {row['merged_claim_text']}",
        f"State the main solution provided by: {row['summary_text_y']} {row['merged_claim_text']}",
        f"Solution from this patent claim: {row['summary_text_y']} {row['merged_claim_text']}",
        f"Main technical approach in: {row['summary_text_y']} {row['merged_claim_text']}",
        f"How does this patent solve the problem? {row['summary_text_y']} {row['merged_claim_text']}",
        f"Proposed method in: {row['summary_text_y']} {row['merged_claim_text']}",
        f"Key solution approach from: {row['summary_text_y']} {row['merged_claim_text']}",
        f"identify solution {row['summary_text_y']} {row['merged_claim_text']}",
        f"solution {row['summary_text_y']} {row['merged_claim_text']}",
    ]

    # # Minimal format variations (problem only)
    # minimal_problem_variations = [
    #     f"Problem: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Problem addressed by this patent: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Main technical challenge solved by: {row['summary_text_y']} {row['merged_claim_text']}"
    # ]
    #
    # # Minimal format variations (solution only)
    # minimal_solution_variations = [
    #     f"Solution: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Solution proposed in this patent: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Key technical approach from: {row['summary_text_y']} {row['merged_claim_text']}"
    # ]
    #
    # # Combined variations (both problem and solution)
    # combined_variations = [
    #     f"Analyze the technical issue and solution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Identify the main technical problem and its solution from: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Identify core problem and solution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Extract key problem and solution from: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"State the key issue and proposed solution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"What is the problem and solution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Identify the problem and solution from: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Problem addressed and solution provided by: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Technical challenge and its solution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Main issue and proposed fix in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Core problem and solution from: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Key technical problem and its resolution in: {row['summary_text_y']} {row['merged_claim_text']}",
    #     f"Primary issue and solution in: {row['summary_text_y']} {row['merged_claim_text']}"
    # ]

    # Randomly select instruction type
    all_variations = [
        # problem_variations,
        solution_variations,
        # minimal_problem_variations,
        # minimal_solution_variations,
        # combined_variations
    ]

    selected_instruction_group = random.choice(all_variations)
    selected_instruction = random.choice(selected_instruction_group)

    # Determine what to include in response based on instruction content
    response_content = []

    instr_lower = selected_instruction.lower()
    response_content = []

    # 1. PRIORITY: Check if instruction asks for BOTH (Combined)
    # This looks for pairs of words that indicate a full analysis
    # is_combined = (
    #         ('problem' in instr_lower or 'issue' in instr_lower or 'challenge' in instr_lower) and
    #         ('solution' in instr_lower or 'resolution' in instr_lower or 'fix' in instr_lower)
    # )
    #
    # if is_combined:
    #     response_content.append(f"Problem: {row['problem']}")
    #     response_content.append(f"Problem Verbatim: {row['problem_verbatim']}")
    #     response_content.append(f"Solution: {row['solution']}")
    #     response_content.append(f"Solution Verbatim: {row['solution_verbatim']}")
    #
    # # 2. Check if instruction asks ONLY for the problem
    # elif any(phrase in instr_lower for phrase in ['problem', 'technical challenge', 'issue']):
    #     response_content.append(f"Problem: {row['problem']}")
    #     response_content.append(f"Problem Verbatim: {row['problem_verbatim']}")
    #
    # # 3. Check if instruction asks ONLY for the solution
    # elif any(phrase in instr_lower for phrase in ['solution', 'technical approach', 'method', 'proposed', 'fix']):
    #     response_content.append(f"Solution: {row['solution']}")
    #     response_content.append(f"Solution Verbatim: {row['solution_verbatim']}")
    #
    # else:
    #     # Default fallback
    #     response_content.append(f"Problem: {row['problem']}")
    #     response_content.append(f"Solution: {row['solution']}")

    # response_content.append(f"Problem: {row['problem']}")
    # response_content.append(f"Problem Verbatim: {row['problem_verbatim']}")
    response_content.append(f"Solution: {row['solution']}")
    response_content.append(f"Solution Verbatim: {row['solution_verbatim']}")
    response = "\n".join(response_content)

    # Format the final output
    messages = [
        {"role": "user", "content": selected_instruction},
        {"role": "assistant", "content": response}
    ]

    # FIX: Return the dictionary with the key "messages"
    return {"messages": messages}


dataset = Dataset.from_pandas(df_cleaned)
dataset = dataset.map(convert_example, remove_columns=dataset.column_names)
ds_split = dataset.train_test_split(test_size=0.1)
train_ds = ds_split["train"]
eval_ds = ds_split["test"]

print(f"Train dataset size: {len(train_ds)}")
print(f"Eval dataset size: {len(eval_ds)}")

###############################################################
# QLORA CONFIGURATION (NEW: The Memory Fix)
###############################################################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

###############################################################
# LOAD MODEL (Gemma 3 270M IT) with QLORA
###############################################################
model_name = "google/gemma-3-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 6000  # Keeping your requested large context

# NEW: Pass quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,  # <--- THIS ENABLES 4-BIT
    device_map="auto",
    # attn_implementation="eager" # Uncomment if you see flash attention warnings
)

# NEW: Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

###############################################################
# LORA CONFIG
###############################################################
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.02,
    bias="none",
    task_type="CAUSAL_LM"  # Explicitly good practice to add
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

###############################################################
# TRAINING ARGS — OPTIMIZED FOR QLORA & 7000 TOKENS
###############################################################
training_args = TrainingArguments(
    output_dir="outputs_qlora_6000_tokens",

    # BATCHING: 7000 tokens is HUGE. We must be conservative even with QLoRA.
    # 4 (batch) * 16 (accum) = 64 Effective Batch Size
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,

    # OPTIMIZER: Paged optimizer offloads to CPU if GPU gets full
    optim="paged_adamw_8bit",

    num_train_epochs=4,
    learning_rate=1e-4,

    warmup_steps=20,
    logging_steps=5,

    # EVALUATION
    eval_strategy="steps",
    eval_steps=20,
    per_device_eval_batch_size=1,

    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=True,

    fp16=False,
    bf16=True,

    max_grad_norm=1.0,
    report_to="none",

    dataloader_num_workers=0,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    disable_tqdm=True,

    gradient_checkpointing=True,
    # Helps group similar length sequences to minimize padding (saves VRAM)
    group_by_length=True,
)

###############################################################
# TRAINER
###############################################################
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    # dataset_text_field="text",
    # max_seq_length=7000,  # Enforcing your limit
)

# QLoRA Stability Fix: Cast norms to float32
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

###############################################################
# TRAIN WITH MEMORY MANAGEMENT
###############################################################

print(f"Starting QLoRA training with 7000 token context window...")
print(f"Train samples: {len(train_ds)}")
print(f"Eval samples: {len(eval_ds)}")

try:
    trainer.train()
    trainer.save_model("final_qlora_adapter")

except Exception as e:
    print(f"Training error: {e}")

    # Fallback if 7000 tokens is still too much for Batch=4
    print("Attempting final fallback (Batch Size=1, Accumulation=64)...")

    training_args_final = TrainingArguments(
        **training_args.to_dict(),
        per_device_train_batch_size=1,  # Minimum possible
        gradient_accumulation_steps=64,  # Maintain effective batch size
        eval_strategy="no",  # Disable eval to save memory
        output_dir="outputs_qlora_fallback",
    )

    trainer_final = SFTTrainer(
        model=model,
        args=training_args_final,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        # dataset_text_field="text",
        # max_seq_length=7000,
    )

    # Re-apply norm casting for the new trainer instance
    for name, module in trainer_final.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print("Running final fallback...")
    trainer_final.train()