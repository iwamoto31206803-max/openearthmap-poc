from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

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

PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 255, 36],
        [148, 148, 148],
        [255, 255, 255],
        [34, 97, 38],
        [0, 69, 255],
        [75, 181, 73],
        [222, 31, 7],
    ],
    dtype=np.uint8,
)

INPUT_SIZE = 512
DEVICE = "cpu"


def default_model_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return (
        base_dir.parent
        / "OpenEarthMap-SAR"
        / "src"
        / "Semantic_Segemtation"
        / "pretrained"
        / "RGB_Real_5_u-efficientnet-b4.pth"
    )


def build_model(model_path: Path) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=9,
        activation=None,
        decoder_attention_type="scse",
    )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.to(DEVICE)


def colorize(class_map: np.ndarray) -> Image.Image:
    rgb = PALETTE[class_map]
    return Image.fromarray(rgb, mode="RGB")


def print_class_summary(class_map: np.ndarray) -> None:
    total = class_map.size

    print()
    print("Class summary")
    print("------------------------------")

    for class_id in range(9):
        count = int(np.count_nonzero(class_map == class_id))
        percent = 100.0 * count / total
        print(
            f"{class_id}: {CLASS_NAMES[class_id]:28s} "
            f"{percent:6.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a first RGB land-cover prediction with the OpenEarthMap PoC model."
    )
    parser.add_argument(
        "image",
        help="Path to an RGB input image (PNG, JPEG, TIFF, etc.).",
    )
    parser.add_argument(
        "--model",
        default=str(default_model_path()),
        help="Path to RGB_Real_5_u-efficientnet-b4.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for prediction outputs.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("RGB land-cover prediction")
    print("------------------------------")
    print("Input image :", image_path)
    print("Model file  :", model_path)
    print("Device      :", DEVICE)
    print("Input size  :", f"{INPUT_SIZE} x {INPUT_SIZE}")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    if original_size != (INPUT_SIZE, INPUT_SIZE):
        print(
            "Note        : input image will be resized to "
            f"{INPUT_SIZE} x {INPUT_SIZE} for this first PoC."
        )

    model = build_model(model_path)
    tensor = preprocess(image)

    with torch.inference_mode():
        logits = model(tensor)
        class_map = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    class_index_path = output_dir / f"{image_path.stem}_classes.png"
    Image.fromarray(class_map, mode="L").save(class_index_path)

    color = colorize(class_map)
    color_original_size = color.resize(original_size, Image.Resampling.NEAREST)

    color_path = output_dir / f"{image_path.stem}_prediction.png"
    color_original_size.save(color_path)

    overlay = Image.blend(
        image,
        color_original_size,
        alpha=0.45,
    )
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    overlay.save(overlay_path)

    print_class_summary(class_map)

    print()
    print("------------------------------")
    print("PREDICTION OK")
    print("Class map :", class_index_path)
    print("Color map :", color_path)
    print("Overlay   :", overlay_path)


if __name__ == "__main__":
    main()
