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

TILE_SIZE = 512
DEFAULT_OVERLAP = 128
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


def make_starts(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)

    return starts


def image_to_tensor(tile: np.ndarray) -> torch.Tensor:
    array = tile.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(DEVICE)


def pad_tile(tile: np.ndarray, tile_size: int):
    h, w = tile.shape[:2]

    pad_h = tile_size - h
    pad_w = tile_size - w

    if pad_h == 0 and pad_w == 0:
        return tile, h, w

    padded = np.pad(
        tile,
        (
            (0, pad_h),
            (0, pad_w),
            (0, 0),
        ),
        mode="edge",
    )

    return padded, h, w


def predict_tiled(
    image: Image.Image,
    model: torch.nn.Module,
    overlap: int,
):
    image_np = np.asarray(image, dtype=np.uint8)
    height, width = image_np.shape[:2]

    stride = TILE_SIZE - overlap

    x_starts = make_starts(width, TILE_SIZE, stride)
    y_starts = make_starts(height, TILE_SIZE, stride)

    logits_sum = np.zeros(
        (9, height, width),
        dtype=np.float32,
    )

    counts = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    total_tiles = len(x_starts) * len(y_starts)
    tile_no = 0

    with torch.inference_mode():
        for y in y_starts:
            for x in x_starts:
                tile_no += 1

                y2 = min(y + TILE_SIZE, height)
                x2 = min(x + TILE_SIZE, width)

                tile = image_np[y:y2, x:x2]
                padded, valid_h, valid_w = pad_tile(
                    tile,
                    TILE_SIZE,
                )

                print(
                    f"Tile {tile_no}/{total_tiles}: "
                    f"x={x}:{x2}, y={y}:{y2}"
                )

                tensor = image_to_tensor(padded)
                logits = model(tensor)[0].cpu().numpy()

                logits = logits[:, :valid_h, :valid_w]

                logits_sum[:, y:y2, x:x2] += logits
                counts[y:y2, x:x2] += 1.0

    logits_avg = logits_sum / counts[None, :, :]

    class_map = np.argmax(
        logits_avg,
        axis=0,
    ).astype(np.uint8)

    # Confidence = maximum softmax probability.
    logits_max = np.max(
        logits_avg,
        axis=0,
        keepdims=True,
    )

    exp_logits = np.exp(
        logits_avg - logits_max
    )

    probs = exp_logits / np.sum(
        exp_logits,
        axis=0,
        keepdims=True,
    )

    confidence = np.max(
        probs,
        axis=0,
    ).astype(np.float32)

    return class_map, confidence


def colorize(class_map: np.ndarray) -> Image.Image:
    return Image.fromarray(
        PALETTE[class_map],
        mode="RGB",
    )


def print_class_summary(class_map: np.ndarray):
    total = class_map.size

    print()
    print("Class summary")
    print("------------------------------")

    for class_id in range(9):
        count = int(
            np.count_nonzero(
                class_map == class_id
            )
        )

        percent = 100.0 * count / total

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:28s} "
            f"{percent:6.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run tiled RGB land-cover prediction "
            "without resizing the whole image."
        )
    )

    parser.add_argument(
        "image",
        help="Path to an RGB input image.",
    )

    parser.add_argument(
        "--model",
        default=str(default_model_path()),
        help="Path to RGB_Real_5_u-efficientnet-b4.pth",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Tile overlap in pixels. Default: 128",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs_tiled",
        help="Directory for prediction outputs.",
    )

    args = parser.parse_args()

    if args.overlap < 0 or args.overlap >= TILE_SIZE:
        raise ValueError(
            f"--overlap must be between 0 and {TILE_SIZE - 1}."
        )

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    print("Tiled RGB land-cover prediction")
    print("----------------------------------")
    print("Input image :", image_path)
    print("Image size  :", image.size)
    print("Model file  :", model_path)
    print("Device      :", DEVICE)
    print("Tile size   :", f"{TILE_SIZE} x {TILE_SIZE}")
    print("Overlap     :", args.overlap)
    print()

    model = build_model(model_path)

    class_map, confidence = predict_tiled(
        image,
        model,
        args.overlap,
    )

    class_path = (
        output_dir
        / f"{image_path.stem}_classes_tiled.png"
    )

    prediction_path = (
        output_dir
        / f"{image_path.stem}_prediction_tiled.png"
    )

    overlay_path = (
        output_dir
        / f"{image_path.stem}_overlay_tiled.png"
    )

    confidence_path = (
        output_dir
        / f"{image_path.stem}_confidence_tiled.png"
    )

    Image.fromarray(
        class_map,
        mode="L",
    ).save(class_path)

    color = colorize(class_map)
    color.save(prediction_path)

    overlay = Image.blend(
        image,
        color,
        alpha=0.45,
    )
    overlay.save(overlay_path)

    confidence_u8 = np.clip(
        confidence * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        confidence_u8,
        mode="L",
    ).save(confidence_path)

    print_class_summary(class_map)

    print()
    print("Confidence summary")
    print("------------------------------")
    print(
        f"Mean confidence : "
        f"{float(confidence.mean()):.3f}"
    )
    print(
        f"Min confidence  : "
        f"{float(confidence.min()):.3f}"
    )
    print(
        f"Max confidence  : "
        f"{float(confidence.max()):.3f}"
    )

    print()
    print("----------------------------------")
    print("TILED PREDICTION OK")
    print("Class map  :", class_path)
    print("Color map  :", prediction_path)
    print("Overlay    :", overlay_path)
    print("Confidence :", confidence_path)


if __name__ == "__main__":
    main()
