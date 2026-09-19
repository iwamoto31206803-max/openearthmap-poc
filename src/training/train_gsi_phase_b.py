"""Phase B: add optional weighted Water and Road teachers to Phase A v0.3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import math
from pathlib import Path
import platform
import random
import shutil
import uuid

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import default_model_path
from src.model import MODEL_KWARGS, PREPROCESSING, build_model, rgb_to_tensor
from src.training.base_preservation import build_teacher, validate_distillation
from src.training.gsi_dataset import (
    DEFAULT_SEED, GsiPaddyDataset, Sample, collate_preservation, scan_dataset, split_samples,
)
from src.training.prepare_gsi_labels import IGNORE_INDEX, _png_index, _sha256
from src.training import replay_preservation as replay
from src.training.train_gsi_paddy import (
    freeze_encoder, make_optimizer, resolve_device, save_checkpoint, seed_everything, write_json,
)

PADDY_CLASS = 7
WATER_CLASS = 6
ROAD_CLASS = 4
PADDY_COUNTS = (1286, 1314)
WATER_COUNTS = (692, 558)
ROAD_COUNTS = (1639, 361)
ROAD_PILOT_COUNT = 1028
ROAD_OVERLAP_COUNT = 7


def scan_water_dataset(org_dir: Path, labels_dir: Path):
    """Audit Water labels without changing the Paddy-specific Phase A scanner."""
    images, labels = _png_index(Path(org_dir), "org"), _png_index(Path(labels_dir), "labels")
    if not images or images.keys() != labels.keys():
        raise ValueError("Water PNG pairing failed")
    positive, all_ignore = [], []
    for relative in sorted(images):
        with Image.open(images[relative]) as image:
            width, height = image.size
            image.verify()
        with Image.open(labels[relative]) as image:
            label = np.array(image)
        if (label.ndim != 2 or label.dtype != np.uint8
                or not np.all((label == WATER_CLASS) | (label == IGNORE_INDEX))):
            raise ValueError("Water labels must be single-channel uint8 containing only 6 and 255")
        if label.shape != (height, width):
            raise ValueError(f"Water image size mismatch for {relative}")
        count = int(np.count_nonzero(label == WATER_CLASS))
        sample = Sample(Path(relative).with_suffix("").as_posix(), images[relative], labels[relative],
                        width, height, count, _sha256(images[relative]), _sha256(labels[relative]))
        (positive if count else all_ignore).append(sample)
    return positive, all_ignore


def scan_road_dataset(org_dir: Path, labels_dir: Path):
    """Audit Road labels, accepting only OEM8 Road and ignore."""
    return _scan_positive_source(org_dir, labels_dir, ROAD_CLASS, "Road")


def _scan_positive_source(org_dir: Path, labels_dir: Path, target: int, source: str):
    images, labels = _png_index(Path(org_dir), "org"), _png_index(Path(labels_dir), "labels")
    if not images or images.keys() != labels.keys():
        raise ValueError(f"{source} PNG pairing failed")
    positive, all_ignore = [], []
    for relative in sorted(images):
        with Image.open(images[relative]) as image:
            width, height = image.size
            image.verify()
        with Image.open(labels[relative]) as image:
            label = np.array(image)
        if (label.ndim != 2 or label.dtype != np.uint8
                or not np.all((label == target) | (label == IGNORE_INDEX))):
            raise ValueError(
                f"{source} labels must be single-channel uint8 containing only {target} and 255")
        if label.shape != (height, width):
            raise ValueError(f"{source} image size mismatch for {relative}")
        count = int(np.count_nonzero(label == target))
        sample = Sample(Path(relative).with_suffix("").as_posix(), images[relative], labels[relative],
                        width, height, count, _sha256(images[relative]), _sha256(labels[relative]))
        (positive if count else all_ignore).append(sample)
    return positive, all_ignore


def exclude_source_hash_overlap(road_samples, paddy_samples):
    """Exclude Road candidates whose source pixels are byte-identical to Paddy."""
    paddy_hashes = {sample.image_sha256 for sample in paddy_samples}
    excluded = sorted((sample for sample in road_samples if sample.image_sha256 in paddy_hashes),
                      key=lambda sample: (sample.image_sha256, sample.source_image_id))
    retained = [sample for sample in road_samples if sample.image_sha256 not in paddy_hashes]
    references = [_road_overlap_reference(sample.image_sha256) for sample in excluded]
    return retained, excluded, references


def _road_overlap_reference(image_sha256: str):
    """Return a stable, domain-separated reference without exposing the source hash."""
    anonymous = hashlib.sha256(f"road-overlap:{image_sha256}".encode("ascii")).hexdigest()
    return f"overlap-{anonymous[:16]}"


def road_pilot_samples(train_pool, count=ROAD_PILOT_COUNT, seed=DEFAULT_SEED):
    """Select the fixed, non-repeating Road Pilot subset deterministically."""
    if count < 0 or count > len(train_pool):
        raise ValueError("Road Pilot count exceeds Road train pool")
    indices = random.Random(seed).sample(range(len(train_pool)), count)
    return [train_pool[index] for index in indices]


class WaterDataset(Dataset):
    def __init__(self, samples):
        if any(sample.positive_pixel_count <= 0 for sample in samples):
            raise ValueError("WaterDataset accepts positive-bearing images only")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(sample.label_path) as image:
            label = np.array(image)
        if (label.shape != rgb.shape[:2]
                or not np.all((label == WATER_CLASS) | (label == IGNORE_INDEX))
                or not np.any(label == WATER_CLASS)):
            raise ValueError("Invalid Water positive partial label")
        return rgb_to_tensor(rgb), torch.from_numpy(label.astype(np.int64))


class RoadDataset(WaterDataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(sample.label_path) as image:
            label = np.array(image)
        if (label.shape != rgb.shape[:2]
                or not np.all((label == ROAD_CLASS) | (label == IGNORE_INDEX))
                or not np.any(label == ROAD_CLASS)):
            raise ValueError("Invalid Road positive partial label")
        return rgb_to_tensor(rgb), torch.from_numpy(label.astype(np.int64))


def positive_preservation_losses(student_logits, teacher_logits, labels, image_mask,
                                 target_class, lambda_preserve=1.0, temperature=1.0):
    """CE on one source's positives plus Base KL on that source's unknown pixels."""
    validate_distillation(lambda_preserve, temperature)
    expected = (labels.shape[0], 9, *labels.shape[1:])
    if student_logits.shape != expected or teacher_logits.shape != expected:
        raise ValueError("Expected matching NCHW logits with 9 classes")
    if image_mask.shape != labels.shape or image_mask.dtype != torch.bool:
        raise ValueError("image_mask must be boolean and match labels")
    if not torch.all((labels == target_class) | (labels == IGNORE_INDEX)):
        raise ValueError(f"Labels must contain only {target_class} and 255")
    positive = image_mask & (labels == target_class)
    unknown = image_mask & (labels == IGNORE_INDEX)
    if not positive.any():
        raise ValueError("Positive source batch is all-ignore")
    student = student_logits.permute(0, 2, 3, 1)
    teacher = teacher_logits.detach().permute(0, 2, 3, 1)
    ce = F.cross_entropy(student[positive], labels[positive])
    if unknown.any():
        kl = F.kl_div(F.log_softmax(student[unknown] / temperature, -1),
                      F.softmax(teacher[unknown] / temperature, -1),
                      reduction="none").sum(-1).mean() * temperature**2
    else:
        kl = student.new_zeros(())
    return ce + lambda_preserve * kl, ce, kl


def water_schedule(slot_count: int, water_count: int, seed: int = DEFAULT_SEED):
    """Select deterministic slots that each consume the next Water sample once."""
    if not 0 <= water_count <= slot_count:
        raise ValueError("Water count must be between zero and Paddy slot count")
    return set(random.Random(seed).sample(range(slot_count), water_count))


def _source_forward(model, teacher, batch, device, target, weight, temperature):
    images, labels, mask = (value.to(device) for value in batch)
    with torch.no_grad():
        base = teacher(images)
    logits = model(images)
    loss, ce, kl = positive_preservation_losses(
        logits, base, labels, mask, target, weight, temperature)
    p, u = mask & (labels == target), mask & (labels == IGNORE_INDEX)
    return loss, ce, kl, logits, base, p, u


def run_phase_b_epoch(model, paddy_loader, device, optimizer=None, *, teacher, replay_loader,
                      water_loader, schedule=None, lambda_preserve=1.0, temperature=1.0,
                      alpha=1.0, beta_water=1.0, progress_interval=50,
                      total_steps=None, road_loader=None, beta_road=1.0):
    """Run one Paddy-axis epoch; optional Water never creates another optimizer update."""
    replay.validate_replay(lambda_preserve, temperature, alpha)
    validate_beta_water(beta_water)
    validate_beta_road(beta_road)
    training = optimizer is not None
    model.train(training); model.encoder.eval(); teacher.requires_grad_(False); teacher.eval()
    schedule = schedule or set()
    sources = {name: {"ce": 0., "kl": 0., "p": 0, "u": 0, "agree": 0,
                             "prob": 0., "base_agree": 0}
               for name in ("paddy", "water", "road")}
    replay_sum = replay_pixels = replay_batches = steps = water_batches = road_batches = 0
    replay_iter, water_iter = iter(replay_loader), iter(water_loader)
    road_iter = iter(road_loader) if road_loader is not None else None
    with torch.set_grad_enabled(training):
        for step, paddy_batch in enumerate(paddy_loader):
            if training: optimizer.zero_grad(set_to_none=True)
            parts = []
            p = _source_forward(model, teacher, paddy_batch, device, PADDY_CLASS,
                                lambda_preserve, temperature)
            parts.append(p[0])
            batches = [("paddy", p, PADDY_CLASS)]
            if step in schedule:
                try: water_batch = next(water_iter)
                except StopIteration: raise ValueError("Water schedule exceeds Water loader") from None
                w = _source_forward(model, teacher, water_batch, device, WATER_CLASS,
                                    lambda_preserve, temperature)
                parts.append(beta_water * w[0])
                batches.append(("water", w, WATER_CLASS)); water_batches += 1
            if road_iter is not None:
                try: road_batch = next(road_iter)
                except StopIteration: raise ValueError("Road loader ended before Paddy axis") from None
                road = _source_forward(model, teacher, road_batch, device, ROAD_CLASS,
                                       lambda_preserve, temperature)
                parts.append(beta_road * road[0])
                batches.append(("road", road, ROAD_CLASS)); road_batches += 1
            try:
                replay_batch = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                try: replay_batch = next(replay_iter)
                except StopIteration: raise ValueError("Replay loader is empty") from None
            ri, rl, rm = (value.to(device) for value in replay_batch)
            with torch.no_grad(): rb = teacher(ri)
            rlogits = model(ri)
            rkl = replay.replay_preservation_loss(rlogits, rb, rl, rm, temperature)
            total = sum(parts) + alpha * lambda_preserve * rkl
            if not torch.isfinite(total): raise RuntimeError("Non-finite Phase B loss")
            if training: total.backward(); optimizer.step()
            with torch.no_grad():
                for name, (_, ce, kl, logits, base, positive, unknown), target in batches:
                    values = sources[name]; pc, uc = int(positive.sum()), int(unknown.sum())
                    values["ce"] += ce.item()*pc; values["kl"] += kl.item()*uc
                    values["p"] += pc; values["u"] += uc
                    pred = logits.argmax(1)
                    values["agree"] += int(((pred == target) & positive).sum())
                    values["prob"] += logits.softmax(1)[:, target][positive].double().sum().item()
                    values["base_agree"] += int(((pred == base.argmax(1)) & unknown).sum())
                count = int(rm.sum()); replay_sum += rkl.item()*count; replay_pixels += count
            steps += 1; replay_batches += 1
            if training and progress_interval and steps % progress_interval == 0:
                expected_steps = total_steps if total_steps is not None else "?"
                paddy_ce = sources["paddy"]["ce"] / sources["paddy"]["p"]
                water_ce = (sources["water"]["ce"] / sources["water"]["p"]
                            if sources["water"]["p"] else None)
                replay_kl_running = replay_sum / replay_pixels
                water_text = f"{water_ce:.6f}" if water_ce is not None else "n/a"
                road_ce = (sources["road"]["ce"] / sources["road"]["p"]
                           if sources["road"]["p"] else None)
                road_text = f"{road_ce:.6f}" if road_ce is not None else "n/a"
                road_total = total_steps if road_iter is not None else 0
                print(f"Phase B step {steps}/{expected_steps}; Water {water_batches}/{len(schedule)}; "
                      f"Road {road_batches}/{road_total}; Paddy CE {paddy_ce:.6f}; "
                      f"Water CE {water_text}; Road CE {road_text}; "
                      f"replay KL {replay_kl_running:.6f}")
    if not steps or not sources["paddy"]["p"] or (schedule and water_batches != len(schedule)):
        raise ValueError("Incomplete Phase B epoch")
    def metrics(name, target):
        value = sources[name]
        if not value["p"]: return None
        return {"positive_ce": value["ce"]/value["p"],
                f"class_{target}_agreement": value["agree"]/value["p"],
                f"class_{target}_mean_probability": value["prob"]/value["p"],
                "unknown_preservation_kl": value["kl"]/value["u"] if value["u"] else 0.,
                "unknown_student_base_argmax_agreement": value["base_agree"]/value["u"] if value["u"] else None,
                "positive_pixel_count": value["p"], "unknown_pixel_count": value["u"]}
    pm, wm, road_metrics = metrics("paddy", 7), metrics("water", 6), metrics("road", 4)
    replay_kl = replay_sum/replay_pixels
    loss = pm["positive_ce"] + lambda_preserve*pm["unknown_preservation_kl"] + alpha*lambda_preserve*replay_kl
    if wm: loss += beta_water * (wm["positive_ce"] + lambda_preserve*wm["unknown_preservation_kl"])
    if road_metrics: loss += beta_road * (road_metrics["positive_ce"] + lambda_preserve*road_metrics["unknown_preservation_kl"])
    return {"loss": loss, "paddy": pm, "water": wm, "road": road_metrics,
            "replay": {"preservation_kl": replay_kl, "pixel_count": replay_pixels,
                       "batch_count": replay_batches},
            "optimizer_update_count": steps if training else 0, "logical_step_count": steps,
            "water_step_count": water_batches, "road_step_count": road_batches}


def validate_phase_b(model, paddy_loader, water_loader, replay_loader, device, *, teacher,
                     lambda_preserve=1.0, temperature=1.0, alpha=1.0, beta_water=1.0,
                     road_loader=None, beta_road=1.0):
    """Evaluate every validation sample once and report each source separately."""
    validate_beta_water(beta_water)
    validate_beta_road(beta_road)
    model.eval(); model.encoder.eval(); teacher.requires_grad_(False); teacher.eval()
    def positive(loader, target):
        totals = {"ce":0.,"kl":0.,"p":0,"u":0,"agree":0,"prob":0.,"base_agree":0}
        with torch.no_grad():
            for batch in loader:
                _, ce, kl, logits, base, p, u = _source_forward(
                    model, teacher, batch, device, target, lambda_preserve, temperature)
                pc, uc = int(p.sum()), int(u.sum()); totals["p"] += pc; totals["u"] += uc
                totals["ce"] += ce.item()*pc; totals["kl"] += kl.item()*uc
                pred=logits.argmax(1); totals["agree"] += int(((pred==target)&p).sum())
                totals["prob"] += logits.softmax(1)[:,target][p].double().sum().item()
                totals["base_agree"] += int(((pred==base.argmax(1))&u).sum())
        if not totals["p"]: raise ValueError("Validation source has no positive pixels")
        return {"positive_ce":totals["ce"]/totals["p"], f"class_{target}_agreement":totals["agree"]/totals["p"],
                f"class_{target}_mean_probability":totals["prob"]/totals["p"],
                "unknown_preservation_kl":totals["kl"]/totals["u"] if totals["u"] else 0.,
                "unknown_student_base_argmax_agreement":totals["base_agree"]/totals["u"] if totals["u"] else None,
                "positive_pixel_count":totals["p"],"unknown_pixel_count":totals["u"]}
    replay_sum = pixels = agreed = 0
    with torch.no_grad():
        for images, labels, mask in replay_loader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            base=teacher(images); logits=model(images)
            kl=replay.replay_preservation_loss(logits,base,labels,mask,temperature); count=int(mask.sum())
            replay_sum += kl.item()*count; pixels += count
            agreed += int(((logits.argmax(1)==base.argmax(1))&mask).sum())
    if not pixels: raise ValueError("Replay validation is empty")
    paddy, water = positive(paddy_loader, PADDY_CLASS), positive(water_loader, WATER_CLASS)
    road = positive(road_loader, ROAD_CLASS) if road_loader is not None else None
    rkl=replay_sum/pixels
    loss=sum((paddy["positive_ce"], lambda_preserve*paddy["unknown_preservation_kl"],
              beta_water*(water["positive_ce"]+lambda_preserve*water["unknown_preservation_kl"]),
              alpha*lambda_preserve*rkl))
    if road: loss += beta_road*(road["positive_ce"]+lambda_preserve*road["unknown_preservation_kl"])
    return {"loss":loss,"paddy":paddy,"water":water,"road":road,
            "replay":{"preservation_kl":rkl,"student_base_argmax_agreement":agreed/pixels,"pixel_count":pixels}}


def _audit_manifest(path, category, target):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gsi_category") != category or value.get("oem_class_id") != target or value.get("ignore_index") != 255:
        raise ValueError(f"Prepared manifest must specify {category}, class {target}, ignore 255")
    return value


def validate_beta_water(beta_water):
    """Reject weights that cannot define a finite, non-negative Water objective."""
    if not math.isfinite(beta_water) or beta_water < 0:
        raise ValueError("beta_water must be finite and >= 0")


def validate_beta_road(beta_road):
    if not math.isfinite(beta_road) or beta_road < 0:
        raise ValueError("beta_road must be finite and >= 0")


def train(args):
    if args.seed != 42 or args.train_ratio != .8 or args.batch_size != 1:
        raise ValueError("Phase B v0.1 requires seed=42, train-ratio=0.8, batch-size=1")
    epochs = 1 if args.epochs is None else args.epochs
    if epochs < 1 or args.num_threads < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid training settings")
    replay.validate_replay(args.lambda_preserve, args.temperature, args.alpha_replay)
    validate_beta_water(args.beta_water)
    validate_beta_road(args.beta_road)
    road_enabled = args.road_org_dir is not None or args.road_prepared_dir is not None
    if road_enabled and (args.road_org_dir is None or args.road_prepared_dir is None):
        raise ValueError("Road requires both --road-org-dir and --road-prepared-dir")
    seed_everything(args.seed); torch.set_num_threads(args.num_threads); device = resolve_device(args.device)
    base = args.base_model.resolve(); base_sha = _sha256(base)
    if args.base_sha256 and base_sha != args.base_sha256.lower(): raise ValueError("Base SHA256 mismatch")
    pp, wp = args.prepared_dir/"manifest.json", args.water_prepared_dir/"manifest.json"
    pman, wman = _audit_manifest(pp, "paddy", 7), _audit_manifest(wp, "water", 6)
    paddy, paddy_ignored = scan_dataset(args.org_dir, args.prepared_dir/"labels")
    water, water_ignored = scan_water_dataset(args.water_org_dir, args.water_prepared_dir/"labels")
    for name, prepared, positive, ignored in (("Paddy", pman, paddy, paddy_ignored),
                                                ("Water", wman, water, water_ignored)):
        actual = {"image_count":len(positive)+len(ignored), "false_image_count":len(ignored),
                  "positive_pixel_count":sum(sample.positive_pixel_count for sample in positive),
                  "total_pixel_count":sum(sample.width*sample.height for sample in positive+ignored)}
        if any(prepared.get(key) != value for key, value in actual.items()):
            raise ValueError(f"{name} prepared manifest counts do not match the audited dataset")
    if (len(paddy), len(paddy_ignored)) != PADDY_COUNTS: raise ValueError(f"Expected Paddy counts {PADDY_COUNTS}")
    if (len(water), len(water_ignored)) != WATER_COUNTS: raise ValueError(f"Expected Water counts {WATER_COUNTS}")
    pt, pv = split_samples(paddy, .8, 42); rt, rv = replay.replay_splits(pt, pv, paddy_ignored, 42)
    wt, wv = split_samples(water, .8, 42)
    road_manifest_path = args.road_prepared_dir/"manifest.json" if road_enabled else None
    road_ignored = excluded_road = road_references = road_pool = road_train_pool = road_used = road_unused = road_validation = []
    if road_enabled:
        road_manifest = _audit_manifest(road_manifest_path, "road", ROAD_CLASS)
        road, road_ignored = scan_road_dataset(args.road_org_dir, args.road_prepared_dir/"labels")
        actual = {"image_count":len(road)+len(road_ignored), "false_image_count":len(road_ignored),
                  "positive_pixel_count":sum(sample.positive_pixel_count for sample in road),
                  "total_pixel_count":sum(sample.width*sample.height for sample in road+road_ignored)}
        if any(road_manifest.get(key) != value for key, value in actual.items()):
            raise ValueError("Road prepared manifest counts do not match the audited dataset")
        if (len(road), len(road_ignored)) != ROAD_COUNTS:
            raise ValueError(f"Expected Road counts {ROAD_COUNTS}")
        road_pool, excluded_road, road_references = exclude_source_hash_overlap(
            road, paddy + paddy_ignored)
        if len(excluded_road) != ROAD_OVERLAP_COUNT:
            raise ValueError(f"Expected {ROAD_OVERLAP_COUNT} Paddy/Road source-image overlaps")
        road_train_pool, road_validation = split_samples(road_pool, .8, 42)
        road_used = road_pilot_samples(road_train_pool, ROAD_PILOT_COUNT, 42)
        if len(pt) != len(road_used):
            raise ValueError("Road Pilot must provide exactly one Road sample per Paddy logical step")
        used_ids = {sample.source_image_id for sample in road_used}
        road_unused = [sample for sample in road_train_pool if sample.source_image_id not in used_ids]
    model = build_model(base, device=device); counts = freeze_encoder(model); optimizer = make_optimizer(model, args.learning_rate)
    teacher = build_teacher(base, device); replay.check_model_invariants(model, teacher, optimizer, initial=True)
    if _sha256(base) != base_sha: raise ValueError("Base checkpoint changed during loading")
    timestamp = datetime.now(timezone.utc); run_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")+"_"+uuid.uuid4().hex[:8]
    run_dir = (args.output_dir or Path("training_outputs/gsi_phase_b_v01")).resolve()/run_id
    run_dir.mkdir(parents=True); (run_dir/".gitignore").write_text("*\n"); (run_dir/"checkpoints").mkdir()
    files = {"paddy_train_ids": pt, "paddy_validation_ids": pv, "paddy_replay_train_ids": rt,
             "paddy_replay_validation_ids": rv, "water_positive_train_ids": wt,
             "water_positive_validation_ids": wv}
    if road_enabled:
        files.update(road_positive_train_pool_ids=road_train_pool,
                     road_positive_pilot_ids=road_used,
                     road_positive_unused_train_ids=road_unused,
                     road_positive_validation_ids=road_validation)
    for name, samples in files.items(): write_json(run_dir/(name+".json"), [s.source_image_id for s in samples])
    schedule = water_schedule(len(pt), len(wt), 42)
    if road_enabled and args.beta_water == .5 and args.beta_road == 1.0:
        experiment = "gsi_phase_b_v0.3"
    elif road_enabled:
        experiment = "gsi_phase_b_custom_road_weight"
    elif args.beta_water == 1.0:
        experiment = "gsi_phase_b_v0.1"
    elif args.beta_water == 0.5:
        experiment = "gsi_phase_b_v0.2"
    else:
        experiment = "gsi_phase_b_custom_water_weight"
    manifest = {"schema_version": 4, "experiment": experiment, "model_version": experiment,
        "training_mode": ("phase_b_v0.3_road_pilot" if experiment == "gsi_phase_b_v0.3" else
                          "phase_b_v0.1_compatible" if not road_enabled and args.beta_water == 1.0 else
                          "phase_b_v0.2" if not road_enabled else "phase_b_custom_road_pilot"),
        "status": "running", "run_id": run_id, "timestamp_utc": timestamp.isoformat(),
        "base_model_path": str(base), "base_model_sha256": base_sha,
        "student_initialization_checkpoint_sha256": base_sha, "teacher_checkpoint_sha256": base_sha,
        "original_base_start": True, "git_commit_sha": replay.current_git_commit(), "seed": 42,
        "lambda_preserve": args.lambda_preserve, "temperature": args.temperature, "alpha_replay": args.alpha_replay,
        "beta_water": args.beta_water, "beta_road": args.beta_road if road_enabled else None,
        "epochs": epochs, "batch_size": 1, "learning_rate": args.learning_rate,
        "optimizer": {"name":"AdamW","weight_decay":.01,"betas":[.9,.999],"eps":1e-8},
        "architecture":{"library":"segmentation_models_pytorch","model":"Unet",**MODEL_KWARGS}, "preprocessing":PREPROCESSING,
        "parameter_counts": counts, "paddy_target_class":7, "water_target_class":6,
        "road_target_class":ROAD_CLASS if road_enabled else None,
        "paddy_train_count":len(pt), "paddy_validation_count":len(pv), "paddy_replay_train_count":len(rt),
        "paddy_replay_validation_count":len(rv), "water_positive_train_count":len(wt),
        "water_positive_validation_count":len(wv), "water_all_ignore_count":len(water_ignored),
        "water_all_ignore_usage":"not used for training, replay, or validation",
        "road_source_enabled":road_enabled,
        "road_source_configuration":({"org":"CLI --road-org-dir", "prepared":"CLI --road-prepared-dir"}
                                     if road_enabled else None),
        "road_positive_pool_before_exclusion":len(road_pool)+len(excluded_road),
        "road_positive_pool_after_exclusion":len(road_pool),
        "road_overlap_count":len(excluded_road), "road_overlap_excluded_count":len(excluded_road),
        "road_overlap_sanitized_references":road_references,
        "road_positive_train_pool_count":len(road_train_pool),
        "road_positive_validation_count":len(road_validation),
        "road_pilot_used_count":len(road_used), "road_unused_train_count":len(road_unused),
        "road_all_ignore_count":len(road_ignored),
        "road_all_ignore_usage":"not used for training, replay, or validation" if road_enabled else None,
        "total_optimizer_updates":epochs*len(pt), "epoch_axis":"Paddy positive train split",
        "logical_step_count_per_epoch":len(pt), "road_step_count_per_epoch":len(road_used),
        "water_step_count_per_epoch":len(wt),
        "split_method":"sort IDs, random.Random(seed).shuffle, floor(n*ratio), clamp to [1,n-1]",
        "water_schedule":{"deterministic":True,"seed":42,"method":"random.sample slot set; consume next Water train sample at each selected slot",
                          "slot_count":len(pt),"water_slot_count":len(wt)},
        "loss":"L_paddy + beta_water * optional L_water + beta_road * optional L_road + alpha_replay * L_replay; L_road = Road positive CE(class 4) + lambda_preserve * T^2 * Road unknown KL(Base || Student); one backward and optimizer step per Paddy logical step",
        "prepared_manifest_sha256":{"paddy":_sha256(pp),"water":_sha256(wp),
                                    **({"road":_sha256(road_manifest_path)} if road_enabled else {})},
        "id_files":{}, "epoch_metrics":[], "best_checkpoint":None, "best_checkpoint_sha256":None,
        "best_epoch":None, "best_validation_loss":None,
        "best_criterion":"minimum sum of source-specific validation objectives; first epoch wins ties",
        "versions":{"python":platform.python_version(),"torch":str(torch.__version__),"numpy":np.__version__,
                    "Pillow":version("Pillow"),"segmentation_models_pytorch":version("segmentation-models-pytorch")}}
    for name in files:
        filename=name+".json"; manifest["id_files"][name]={"path":filename,"sha256":_sha256(run_dir/filename)}
    write_json(run_dir/"run_manifest.json", manifest)
    collate = replay.checked_collate
    def loader(dataset, shuffle=False, seed=42):
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                          num_workers=0, collate_fn=collate)
    pl, pvl = loader(GsiPaddyDataset(pt), True), loader(GsiPaddyDataset(pv))
    rl, rvl = loader(replay.GsiReplayDataset(rt), True), loader(replay.GsiReplayDataset(rv))
    wl, wvl = loader(WaterDataset(wt)), loader(WaterDataset(wv))
    road_loader = loader(RoadDataset(road_used)) if road_enabled else None
    road_validation_loader = loader(RoadDataset(road_validation)) if road_enabled else None
    try:
        if args.preflight:
            manifest["preflight_metrics"] = run_phase_b_epoch(model, islice(pl,1), device, teacher=teacher,
                replay_loader=list(islice(rl,1)), water_loader=list(islice(wl,1)), schedule={0},
                lambda_preserve=args.lambda_preserve, temperature=args.temperature, alpha=args.alpha_replay,
                beta_water=args.beta_water, total_steps=1,
                road_loader=list(islice(road_loader,1)) if road_enabled else None, beta_road=args.beta_road)
            manifest["status"]="preflight_passed"; return run_dir
        for epoch in range(1, epochs+1):
            train_schedule = {0} if args.smoke_test else schedule
            training = run_phase_b_epoch(model, islice(pl,1) if args.smoke_test else pl, device, optimizer,
                teacher=teacher,replay_loader=list(islice(rl,1)) if args.smoke_test else rl,
                water_loader=list(islice(wl,1)) if args.smoke_test else wl,schedule=train_schedule,
                lambda_preserve=args.lambda_preserve,temperature=args.temperature,alpha=args.alpha_replay,
                beta_water=args.beta_water,total_steps=1 if args.smoke_test else len(pt),
                road_loader=(list(islice(road_loader,1)) if args.smoke_test and road_enabled else road_loader),
                beta_road=args.beta_road)
            validation = validate_phase_b(model,pvl,wvl,rvl,device,teacher=teacher,
                lambda_preserve=args.lambda_preserve,temperature=args.temperature,alpha=args.alpha_replay,
                beta_water=args.beta_water,road_loader=road_validation_loader,beta_road=args.beta_road)
            relative=f"checkpoints/epoch_{epoch:03d}.pth"; save_checkpoint(run_dir/relative,model)
            manifest["epoch_metrics"].append({"epoch":epoch,"training":training,"validation":validation,"checkpoint":relative})
            if manifest["best_validation_loss"] is None or validation["loss"] < manifest["best_validation_loss"]:
                shutil.copyfile(run_dir/relative,run_dir/"checkpoints/best.pth")
                manifest.update(best_checkpoint="checkpoints/best.pth",best_epoch=epoch,best_validation_loss=validation["loss"],
                                best_checkpoint_sha256=_sha256(run_dir/"checkpoints/best.pth"))
            write_json(run_dir/"run_manifest.json",manifest)
        shutil.copyfile(run_dir/relative,run_dir/"checkpoints/final.pth"); manifest["final_checkpoint"]="checkpoints/final.pth"
        manifest["final_checkpoint_sha256"]=_sha256(run_dir/"checkpoints/final.pth")
        manifest["status"]="smoke_test_completed" if args.smoke_test else "completed"
    except BaseException as exc:
        manifest.update(status="failed",failure_type=type(exc).__name__); raise
    finally:
        manifest["finished_at_utc"]=datetime.now(timezone.utc).isoformat(); write_json(run_dir/"run_manifest.json",manifest)
    return run_dir


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-dir",type=Path,required=True); parser.add_argument("--prepared-dir",type=Path,required=True)
    parser.add_argument("--water-org-dir",type=Path,required=True); parser.add_argument("--water-prepared-dir",type=Path,required=True)
    parser.add_argument("--road-org-dir",type=Path); parser.add_argument("--road-prepared-dir",type=Path)
    parser.add_argument("--base-model",type=Path,default=default_model_path()); parser.add_argument("--base-sha256")
    parser.add_argument("--output-dir",type=Path); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--train-ratio",type=float,default=.8); parser.add_argument("--epochs",type=int,default=None)
    parser.add_argument("--batch-size",type=int,default=1); parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--lambda-preserve",type=float,default=1.); parser.add_argument("--temperature",type=float,default=1.)
    parser.add_argument("--alpha-replay",type=float,default=1.); parser.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    parser.add_argument("--beta-water",type=float,default=1.,help="weight for the complete Water source objective")
    parser.add_argument("--beta-road",type=float,default=1.,help="weight for the complete Road source objective")
    parser.add_argument("--num-threads",type=int,default=2)
    checks=parser.add_mutually_exclusive_group(); checks.add_argument("--preflight",action="store_true"); checks.add_argument("--smoke-test",action="store_true")
    return parser


if __name__ == "__main__": train(make_parser().parse_args())
