from contextlib import nullcontext

import torch


def get_default_device():
    """Return the fastest available PyTorch inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_inference(model, device, fp32=False):
    """Move a model to its inference device and select a fast, safe dtype."""
    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = model.to(device=device, dtype=torch.float32)
    elif device.type == "mps":
        # Some supported PyTorch builds do not implement MPS autocast.
        # Explicit FP16 also avoids the memory cost of a full FP32 window.
        dtype = torch.float32 if fp32 else torch.float16
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=torch.float32)

    return model.eval()


def inference_context(device, fp32=False):
    """Inference-mode context with autocast only where it is supported."""
    device = torch.device(device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and not fp32
        else nullcontext()
    )
    return torch.inference_mode(), autocast


def synchronize(device):
    """Synchronize an accelerator before recording a timing."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
