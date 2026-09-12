"""Shared PoC configuration.

Keep model-independent land-cover metadata and provisional PoC defaults here.
Model file paths are resolved separately so that the model can later be swapped.
"""

from pathlib import Path

POC_VERSION = "0.1"

CLASS_NAMES = {
    0: "Background / Unlabelled",
    1: "Bareland",
    2: "Grass / Rangeland",
    3: "Pavement / Developed space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Cropland / Agriculture",
    8: "Buildings",
}

CLASS_COLORS = {
    0: (0, 0, 0, 255),
    1: (128, 0, 0, 255),
    2: (0, 255, 36, 255),
    3: (148, 148, 148, 255),
    4: (255, 255, 255, 255),
    5: (34, 97, 38, 255),
    6: (0, 69, 255, 255),
    7: (75, 181, 73, 255),
    8: (222, 31, 7, 255),
}

MODEL_FILENAME = "RGB_Real_5_u-efficientnet-b4.pth"
MODEL_ARCHITECTURE = "Unet / EfficientNet-B4 / scSE"
MODEL_CLASSES = 9

INFERENCE_TILE_SIZE = 512
DEFAULT_OVERLAP = 128
DEFAULT_SIEVE_AREA_M2 = 5.0
DEFAULT_SIEVE_CONNECTIVITY = 8

GSI_TILE_SIZE = 256
GSI_TILE_URL = (
    "https://cyberjapandata.gsi.go.jp/xyz/"
    "seamlessphoto/{z}/{x}/{y}.jpg"
)
GSI_SOURCE_NAME = "GSI Tiles / Seamless Aerial Photography"
GSI_SOURCE_NAME_JP = "国土地理院 全国最新写真（シームレス）"


def repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def workspace_dir() -> Path:
    """Directory containing this repo; OpenEarthMap-SAR is expected as a sibling."""
    return repo_dir().parent


def default_model_path() -> Path:
    return (
        workspace_dir()
        / "OpenEarthMap-SAR"
        / "src"
        / "Semantic_Segemtation"
        / "pretrained"
        / MODEL_FILENAME
    )
