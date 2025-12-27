import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MERGED_MODEL_PATH = "merged_gemma_for_gguf"  # Should match your EXPORT_DIR
TEST_PROMPT = "What are the three most important steps for successful QLoRA fine-tuning on Gemma?"


def test_inference_with_template(model_path, prompt):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 1. Ensure padding is correct for Gemma (usually set during training, but good to check)
        tokenizer.pad_token = tokenizer.eos_token

        # 2. Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 3. Create the chat template for the input
        messages = [{"role": "user", "content": prompt}]
        # This function generates the exact <start_of_turn>user...<end_of_turn><start_of_turn>model format
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Tells the model it's time for the assistant to respond
        )

        print("\n--- Formatted Prompt (Model Input) ---")
        print(repr(formatted_prompt))  # Use repr to see special tokens
        print("--------------------------------------")

        # 4. Generate the response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id  # Crucial for stopping generation
        )

        # Decode only the newly generated part
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print("\n--- Model Output (Unformatted) ---")
        print(generated_text.strip())
        print("----------------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        print("Ensure your merged model directory is correct and the necessary libraries are installed.")


test_inference_with_template(MERGED_MODEL_PATH, TEST_PROMPT)