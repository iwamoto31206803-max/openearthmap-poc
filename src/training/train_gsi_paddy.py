"""Phase A pilot: frozen encoder, GSI paddy class-7 positive-only partial labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, default_model_path
from src.model import MODEL_KWARGS, PREPROCESSING, build_model
from src.training.gsi_dataset import (
    DEFAULT_SEED, TARGET_CLASS, GsiPaddyDataset, collate_padded, scan_dataset,
    split_samples,
)
from src.training.prepare_gsi_labels import IGNORE_INDEX, _sha256


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def freeze_encoder(model: torch.nn.Module) -> dict[str, int]:
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.segmentation_head.requires_grad_(True)
    model.encoder.eval()  # Also freeze BN buffers and encoder dropout behavior.
    return {
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def make_optimizer(model: torch.nn.Module, learning_rate: float):
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, weight_decay=0.01,
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    model.encoder.eval()  # model.train() would otherwise update frozen BN buffers.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss_sum, labeled, agreed, probability_sum = 0.0, 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            valid = labels != IGNORE_INDEX
            count = int(valid.sum().item())
            if count == 0:
                raise ValueError("All-ignore batch has no training/validation signal")
            if not torch.all(labels[valid] == TARGET_CLASS):
                raise ValueError("Only class-7 positive partial labels are supported")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; aborting run")
            if training:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                loss_sum += loss.item() * count
                labeled += count
                agreed += int(((logits.argmax(1) == TARGET_CLASS) & valid).sum().item())
                probability_sum += float(
                    logits.softmax(1)[:, TARGET_CLASS][valid].double().sum().item()
                )
            if batch_index % 50 == 0:
                phase = "train" if training else "validation"
                print(f"  {phase}: {batch_index} batches, {labeled:,} labeled pixels, "
                      f"loss={loss_sum / labeled:.6f}", flush=True)
    if labeled == 0:
        raise ValueError("No labeled pixels in epoch")
    return {
        "loss": loss_sum / labeled,
        "labeled_pixel_count": labeled,
        "class_7_labeled_pixel_agreement_recall": agreed / labeled,
        "class_7_mean_probability_on_labeled_pixels": probability_sum / labeled,
    }


def save_checkpoint(path: Path, model) -> None:
    """Plain CPU state_dict; metadata is kept separately in run_manifest.json."""
    temporary = path.with_suffix(".pth.tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    return value


def train(args) -> Path:
    if args.epochs < 1 or args.batch_size < 1 or args.num_threads < 1:
        raise ValueError("epochs, batch-size and num-threads must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be between 0 and 1 (exclusive)")
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    base_path = args.base_model.resolve()
    base_sha = _sha256(base_path)
    prepared_path = args.prepared_dir / "manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (prepared.get("gsi_category") != "paddy"
            or prepared.get("oem_class_id") != TARGET_CLASS
            or prepared.get("ignore_index") != IGNORE_INDEX):
        raise ValueError("Prepared manifest must specify paddy, oem_class_id=7, ignore_index=255")
    usable, excluded = scan_dataset(args.org_dir, args.prepared_dir / "labels")
    # Verify supplied prepared audit against actual labels before training.
    actual = {
        "image_count": len(usable) + len(excluded),
        "false_image_count": len(excluded),
        "positive_pixel_count": sum(s.positive_pixel_count for s in usable),
        "total_pixel_count": sum(s.width * s.height for s in usable + excluded),
    }
    if any(prepared.get(key) != count for key, count in actual.items()):
        raise ValueError("Prepared manifest counts do not match current dataset; regenerate/audit labels")
    train_samples, validation_samples = split_samples(usable, args.train_ratio, args.seed)
    model = build_model(base_path, device=device)
    counts = freeze_encoder(model)
    optimizer = make_optimizer(model, args.learning_rate)
    print(f"Device: {device}; frozen parameters: {counts['frozen']:,}; "
          f"trainable parameters: {counts['trainable']:,}", flush=True)
    print(f"Images: {actual['image_count']}; positive: {len(usable)}; "
          f"excluded all-ignore: {len(excluded)}; train/validation: "
          f"{len(train_samples)}/{len(validation_samples)}", flush=True)

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = args.output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # Protect outputs even when a caller chooses a custom directory inside a repo.
    (run_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    (run_dir / "checkpoints").mkdir()
    write_json(run_dir / "train_ids.json", [s.source_image_id for s in train_samples])
    write_json(run_dir / "validation_ids.json", [s.source_image_id for s in validation_samples])
    write_json(run_dir / "excluded_ids.json", {
        "reason": "all pixels are 255 (unknown/ignore); no supervised loss signal",
        "source_image_ids": [s.source_image_id for s in excluded],
    })
    # Only IDs and image content hashes; no input directories, geospatial fields,
    # or arbitrary prepared manifest fields are copied into this inventory.
    inventory = []
    for sample in sorted(usable + excluded, key=lambda s: s.source_image_id):
        row = asdict(sample)
        del row["image_path"], row["label_path"]
        inventory.append(row)
    write_json(run_dir / "dataset_inventory.json", inventory)
    manifest = {
        "schema_version": 1, "run_id": run_id, "timestamp_utc": timestamp.isoformat(),
        "status": "running", "base_model_path": str(base_path), "base_model_sha256": base_sha,
        "architecture": {"library": "segmentation_models_pytorch", "model": "Unet", **MODEL_KWARGS},
        "preprocessing": PREPROCESSING,
        "padding": "right/bottom to batch maximum multiple of 32; RGB=edge, label=255",
        "augmentation": "none", "gsi_category": "paddy", "oem_target_class": TARGET_CLASS,
        "oem_target_class_name": CLASS_NAMES[TARGET_CLASS],
        "prepared_manifest_sha256": _sha256(prepared_path),
        "dataset_inventory": "dataset_inventory.json",
        "dataset_inventory_sha256": _sha256(run_dir / "dataset_inventory.json"),
        "total_images": actual["image_count"], "usable_positive_images": len(usable),
        "excluded_all_ignore_images": len(excluded),
        "exclusion_reason": "all pixels are 255; no supervised loss signal",
        "excluded_image_ids": "excluded_ids.json",
        "train_count": len(train_samples), "validation_count": len(validation_samples),
        "split_seed": args.seed, "train_ratio_requested": args.train_ratio,
        "train_ratio_actual": len(train_samples) / len(usable),
        "split_method": "sort IDs, random.Random(seed).shuffle, floor(n*ratio), clamp to [1,n-1]",
        "train_image_ids": "train_ids.json", "validation_image_ids": "validation_ids.json",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "learning_rate": args.learning_rate, "loss": "CrossEntropyLoss(ignore_index=255, reduction='mean')",
        "epoch_loss_aggregation": "weighted by labeled pixel count across batches",
        "ignore_index": IGNORE_INDEX, "parameter_counts": counts,
        "frozen_modules": ["encoder (parameters and BatchNorm statistics)"],
        "trainable_modules": ["decoder", "segmentation_head"], "device": device,
        "num_workers": 0, "num_threads": args.num_threads, "deterministic_algorithms": True,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "segmentation_models_pytorch": version("segmentation-models-pytorch"),
                     "numpy": np.__version__, "Pillow": version("Pillow")},
        "platform": platform.platform(),
        "epoch_metrics": [], "best_checkpoint": None, "best_epoch": None,
        "best_validation_loss": None, "best_criterion": "minimum validation loss; first epoch wins ties",
        "final_checkpoint": None, "checkpoint_format": "plain CPU state_dict, inference-compatible; no optimizer/resume state",
        "known_limitations": [
            "Not a spatially independent split; image-level locations are unavailable.",
            "Class-7 positive-only partial supervision; unknown pixels are not negatives.",
            "Validation measures partial-label agreement, not overall model accuracy.",
            "Other-class degradation and class-7 overprediction cannot be assessed here.",
            "Exact numerical reproducibility across hardware/library versions is not guaranteed.",
        ],
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Run output: {run_dir}", flush=True)
    train_loader = DataLoader(
        GsiPaddyDataset(train_samples), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        collate_fn=collate_padded,
    )
    validation_loader = DataLoader(
        GsiPaddyDataset(validation_samples), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_padded,
    )
    try:
        for epoch in range(1, args.epochs + 1):
            training = run_epoch(model, train_loader, device, optimizer)
            validation = run_epoch(model, validation_loader, device)
            relative = f"checkpoints/epoch_{epoch:03d}.pth"
            save_checkpoint(run_dir / relative, model)
            metric = {"epoch": epoch, "training": training, "validation": validation,
                      "checkpoint": relative}
            manifest["epoch_metrics"].append(metric)
            if (manifest["best_validation_loss"] is None
                    or validation["loss"] < manifest["best_validation_loss"]):
                shutil.copyfile(run_dir / relative, run_dir / "checkpoints/best.pth")
                manifest.update(best_checkpoint="checkpoints/best.pth", best_epoch=epoch,
                                best_validation_loss=validation["loss"])
            write_json(manifest_path, manifest)
            print(f"Epoch {epoch}/{args.epochs}: train loss={training['loss']:.6f}, "
                  f"validation loss={validation['loss']:.6f}, labeled pixels="
                  f"{validation['labeled_pixel_count']}, class-7 agreement="
                  f"{validation['class_7_labeled_pixel_agreement_recall']:.6f}", flush=True)
        shutil.copyfile(run_dir / relative, run_dir / "checkpoints/final.pth")
        manifest.update(final_checkpoint="checkpoints/final.pth", status="completed")
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    return run_dir


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-dir", type=Path, required=True, help="GSI raw paddy_572/org directory")
    parser.add_argument("--prepared-dir", type=Path, required=True, help="contains labels/ and manifest.json")
    parser.add_argument("--base-model", type=Path, default=default_model_path())
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gsi_phase_a"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=2, help="CPU intra-op threads (default: 2)")
    return parser


def main():
    train(make_parser().parse_args())


if __name__ == "__main__":
    main()
