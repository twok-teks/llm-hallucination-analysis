from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Running forward pass...")

    with torch.no_grad():
        outputs = model(**inputs)

    print("Success!")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Number of hidden states tensors: {len(outputs.hidden_states)}")
    print(f"Last hidden states shape: {outputs.hidden_states[-1].shape}")

if __name__ == "__main__":
    main()