from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Model loaded on: {device}")
    return tokenizer, model, device


if __name__ == "__main__":
    tokenizer, model, device = load_model()
    print("Tokenizer and model loaded successfully.")