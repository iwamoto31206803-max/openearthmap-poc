"""Model construction and RGB preprocessing shared by inference and training."""

from pathlib import Path

import numpy as np
import torch

MODEL_KWARGS = {
    "encoder_name": "efficientnet-b4",
    "encoder_weights": None,
    "in_channels": 3,
    "classes": 9,
    "activation": None,
    "decoder_attention_type": "scse",
}
PREPROCESSING = "uint8 RGB -> float32 / 255.0 -> CHW; no mean/std normalization"


def build_model(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Load either a plain state_dict or the existing state_dict wrapper strictly."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    import segmentation_models_pytorch as smp

    model = smp.Unet(**MODEL_KWARGS)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with the existing inference loader.
        checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))
