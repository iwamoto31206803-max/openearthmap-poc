from pathlib import Path
import argparse

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_OVERLAP,
    INFERENCE_TILE_SIZE,
    default_model_path,
)

TILE_SIZE = INFERENCE_TILE_SIZE
DEVICE = "cpu"

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


def pad_tile(tile: np.ndarray, tile_size: int):
    h, w = tile.shape[:2]
    pad_h = tile_size - h
    pad_w = tile_size - w

    if pad_h == 0 and pad_w == 0:
        return tile, h, w

    padded = np.pad(
        tile,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )
    return padded, h, w


def image_to_tensor(tile: np.ndarray) -> torch.Tensor:
    array = tile.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(DEVICE)


def predict_tiled(image: np.ndarray, model: torch.nn.Module, overlap: int):
    height, width = image.shape[:2]
    stride = TILE_SIZE - overlap

    x_starts = make_starts(width, TILE_SIZE, stride)
    y_starts = make_starts(height, TILE_SIZE, stride)

    logits_sum = np.zeros((9, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    total_tiles = len(x_starts) * len(y_starts)
    tile_no = 0

    with torch.inference_mode():
        for y in y_starts:
            for x in x_starts:
                tile_no += 1
                y2 = min(y + TILE_SIZE, height)
                x2 = min(x + TILE_SIZE, width)

                tile = image[y:y2, x:x2]
                padded, valid_h, valid_w = pad_tile(tile, TILE_SIZE)

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
    class_map = np.argmax(logits_avg, axis=0).astype(np.uint8)

    logits_max = np.max(logits_avg, axis=0, keepdims=True)
    exp_logits = np.exp(logits_avg - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    confidence = np.max(probs, axis=0).astype(np.float32)

    return class_map, confidence


def print_class_summary(class_map: np.ndarray):
    total = class_map.size
    print()
    print("Class summary")
    print("------------------------------")

    for class_id in range(9):
        count = int(np.count_nonzero(class_map == class_id))
        percent = 100.0 * count / total
        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:28s} "
            f"{percent:6.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run tiled land-cover prediction on an RGB GeoTIFF "
            "and preserve its CRS and transform."
        )
    )
    parser.add_argument("image", help="Path to a 3-band RGB GeoTIFF.")
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
        default="outputs_geotiff",
        help="Directory for GeoTIFF outputs.",
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
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError("Input GeoTIFF must have at least 3 bands (RGB).")

        rgb = src.read([1, 2, 3])
        rgb = np.moveaxis(rgb, 0, -1).astype(np.uint8)

        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    print("GeoTIFF tiled land-cover prediction")
    print("-------------------------------------")
    print("Input image :", image_path)
    print("Image size  :", f"{width} x {height}")
    print("CRS         :", crs)
    print("Model file  :", model_path)
    print("Device      :", DEVICE)
    print("Tile size   :", f"{TILE_SIZE} x {TILE_SIZE}")
    print("Overlap     :", args.overlap)
    print()

    model = build_model(model_path)
    class_map, confidence = predict_tiled(
        rgb,
        model,
        args.overlap,
    )

    class_path = output_dir / f"{image_path.stem}_classes.tif"
    confidence_path = output_dir / f"{image_path.stem}_confidence.tif"

    class_profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(class_map, 1)
        dst.set_band_description(1, "Land-cover class ID")
        dst.write_colormap(1, CLASS_COLORS)
        dst.update_tags(
            model="RGB_Real_5_u-efficientnet-b4.pth",
            class_0=CLASS_NAMES[0],
            class_1=CLASS_NAMES[1],
            class_2=CLASS_NAMES[2],
            class_3=CLASS_NAMES[3],
            class_4=CLASS_NAMES[4],
            class_5=CLASS_NAMES[5],
            class_6=CLASS_NAMES[6],
            class_7=CLASS_NAMES[7],
            class_8=CLASS_NAMES[8],
        )

    confidence_profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(
        confidence_path,
        "w",
        **confidence_profile,
    ) as dst:
        dst.write(confidence, 1)
        dst.set_band_description(1, "Maximum softmax probability")
        dst.update_tags(
            units="0-1",
            description="Model confidence, not empirical accuracy",
        )

    print_class_summary(class_map)

    print()
    print("Confidence summary")
    print("------------------------------")
    print(f"Mean confidence : {float(confidence.mean()):.3f}")
    print(f"Min confidence  : {float(confidence.min()):.3f}")
    print(f"Max confidence  : {float(confidence.max()):.3f}")

    print()
    print("-------------------------------------")
    print("GEOTIFF PREDICTION OK")
    print("Class GeoTIFF      :", class_path)
    print("Confidence GeoTIFF :", confidence_path)
    print("CRS preserved      :", crs)


if __name__ == "__main__":
    main()
