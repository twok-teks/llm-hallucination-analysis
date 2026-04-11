from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
}


def get_model_name(model_key: str) -> str:
    key = model_key.lower().strip()
    if key not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model key '{model_key}'. Valid options: {valid}")
    return MODEL_REGISTRY[key]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str):
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_model(model_key: str):
    model_name = get_model_name(model_key)
    device = pick_device()
    dtype = pick_dtype(device)

    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model for: {model_name}")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model.to(device)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {device}")
    return tokenizer, model, device, model_name


if __name__ == "__main__":
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for key, value in MODEL_REGISTRY.items():
        print(f"{key} -> {value}")