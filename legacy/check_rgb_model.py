from pathlib import Path
import torch
import segmentation_models_pytorch as smp

# --------------------------------------------------
# Settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "pretrained"
    / "RGB_Real_5_u-efficientnet-b4.pth"
)

DEVICE = "cpu"

# --------------------------------------------------
# Check model file
# --------------------------------------------------

print("RGB model check")
print("------------------------------")
print("Model file:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print(
    "Model size:",
    round(MODEL_PATH.stat().st_size / 1024 / 1024, 1),
    "MB",
)

# --------------------------------------------------
# Build model
# 0 = background/unlabelled
# 1-8 = land-cover classes
# --------------------------------------------------

model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=9,
    activation=None,
    decoder_attention_type="scse",
)

# --------------------------------------------------
# Load pretrained weights
# --------------------------------------------------

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
except TypeError:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

# Also support checkpoints that wrap a state_dict
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("------------------------------")
print("RGB MODEL LOAD OK")
print("Input channels : 3")
print("Output classes : 9")
print("Encoder        : EfficientNet-B4")
print("Device         :", DEVICE)
