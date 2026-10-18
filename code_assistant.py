"""
Interactive Local Coding Assistant powered by on-device GPU inference.
Uses your physical RTX 3050 Laptop GPU to fix bugs and synthesize code locally.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

def load_assistant():
    print("\n[NEUROCODE] Initializing local model on RTX 3050 Laptop GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"[NEUROCODE] Ready! Model active in VRAM: {vram_mb:.1f} MB (Total VRAM: 4.0 GB)\n")
    return model, tokenizer

def main():
    model, tokenizer = load_assistant()
    print("=" * 65)
    print("      NEUROCODE LOCAL OFFLINE CODING COPILOT (RTX 3050)          ")
    print("      Type your coding task or bug. Type 'exit' to quit.         ")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break

            prompt = f"""You are an expert autonomous software engineer.
Task: {user_input}

Provide clean, minimal, syntactically correct Python code with an explanation:
"""
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print("\nNeuroCode Copilot:\n" + "-" * 50)
            print(response.strip())
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting NeuroCode Copilot. Happy coding!")
            break

if __name__ == "__main__":
    main()
