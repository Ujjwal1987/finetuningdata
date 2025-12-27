import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
# The base directory containing all your checkpoint folders
BASE_OUTPUT_DIR = "final_qlora_adapter"

# # 🚨 CRITICAL FIX: The LoRA adapter config is usually inside a checkpoint folder.
# # We identified the lowest eval_loss (1.8101) occurred around Epoch 1.927,
# # which corresponds to approximately step 2000.
# CHECKPOINT_DIR = "checkpoint-200"
LORA_PATH = os.path.join(BASE_OUTPUT_DIR)

MODEL_NAME = "google/gemma-3-270m-it"  # Base model name
EXPORT_DIR = "merged_gemma_for_gguf"  # New directory for the merged model

print(f"Attempting to load LoRA adapter from: {LORA_PATH}")


def merge_and_save_for_export(base_model_id, lora_path, export_path):
    """
    Loads the base model, applies the fine-tuned LoRA adapter, merges the weights,
    and saves the resulting full model to a new directory for GGUF conversion.
    """
    # 1. Load Tokenizer (from the BASE model ID for robustness against missing config in LoRA folder)
    try:
        print(f"Loading tokenizer from base model ID: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Load Base Model
    print(f"Loading base model: {base_model_id}")
    try:
        # Load the base model in a compatible dtype (bf16 for Gemma)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 3. Load Peft/LoRA Adapter - This now uses the LORA_PATH constructed from the checkpoint directory.
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 4. CRITICAL STEP: Merge the LoRA weights into the base model weights
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to: {export_path}")

    # 5. Save the merged full model and tokenizer to a new directory
    merged_model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    print("--- MERGE COMPLETE ---")
    print(f"The merged model is now ready for GGUF conversion in the '{export_path}' directory.")


# Execute the merging process
if os.path.exists(LORA_PATH):
    merge_and_save_for_export(MODEL_NAME, LORA_PATH, EXPORT_DIR)
else:
    print(f"Error: LoRA Checkpoint path '{LORA_PATH}' not found.")
    print("Please ensure you have updated the 'CHECKPOINT_DIR' variable correctly and the folder exists.")