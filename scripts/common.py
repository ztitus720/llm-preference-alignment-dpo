"""Shared helpers: device / dtype selection and chat-formatted tokenisation."""
import torch


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str | None = None):
    """
    bf16 only on hardware that actually has bf16 units — compute capability
    8.0 (Ampere) and up.

    NOT `torch.cuda.is_bf16_supported()`: recent PyTorch returns True for it on
    a T4, because it counts slow software emulation as support. A T4 is Turing
    (sm_75) and must run fp16.
    """
    device = device or pick_device()
    if device != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def dtype_flags(dtype) -> dict:
    """Trainer flags matching the chosen dtype."""
    return {"bf16": dtype is torch.bfloat16, "fp16": dtype is torch.float16}


def load_model(model_name: str, dtype=None, device: str | None = None):
    from transformers import AutoModelForCausalLM

    device = device or pick_device()
    dtype = dtype if dtype is not None else pick_dtype(device)
    try:                                  # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_encode(tok, prompt: str, device: str):
    """
    Chat-template a single user turn into model inputs.

    Returns a dict with input_ids AND attention_mask, both on `device` — the
    original scripts returned bare input_ids on CPU, which either warns about a
    missing attention mask or crashes with a device mismatch once the model is
    on the GPU.
    """
    messages = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return {k: v.to(device) for k, v in enc.items()}
