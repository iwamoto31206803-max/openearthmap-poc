"""Phase A v0.3: all-ignore images preserve Base distributions, never labels."""

import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.model import rgb_to_tensor
from src.training.base_preservation import (
    preservation_losses, run_preservation_epoch, validate_distillation,
)
from src.training.gsi_dataset import TARGET_CLASS, collate_preservation, read_label, split_samples
from src.training.prepare_gsi_labels import IGNORE_INDEX

PILOT_COUNTS = (1286, 1314)


def checked_collate(batch):
    """Use the v0.2 padding implementation and verify original-image extents."""
    images, labels, mask = collate_preservation(batch)
    for index, (_, label) in enumerate(batch):
        height, width = label.shape
        if (not mask[index, :height, :width].all()
                or mask[index, height:].any() or mask[index, :, width:].any()):
            raise ValueError("Padding mask must separate every real pixel from synthetic padding")
    return images, labels, mask


class GsiReplayDataset(Dataset):
    def __init__(self, samples):
        if any(s.positive_pixel_count != 0 for s in samples):
            raise ValueError("Replay requires all-ignore images with zero class-7 pixels")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        label = read_label(sample.label_path)
        if rgb.shape[:2] != label.shape:
            raise ValueError("Replay image/label size mismatch")
        if not np.all(label == IGNORE_INDEX):
            raise ValueError("Replay requires all-ignore labels; class 7 is forbidden")
        return rgb_to_tensor(rgb), torch.from_numpy(label.astype(np.int64))


def replay_splits(positive_train, positive_validation, replay, seed=42):
    # Separate local RNG: never shuffle the positive list or consume its RNG.
    train, validation = split_samples(replay, 0.8, seed)
    groups = [positive_train, positive_validation, train, validation]
    seen = set()
    for samples in groups:
        ids = {s.source_image_id for s in samples}
        if not ids or len(ids) != len(samples) or seen & ids:
            raise ValueError("Positive/replay train/validation IDs must be unique and disjoint")
        seen.update(ids)
    return train, validation


def validate_replay(lambda_preserve, temperature, alpha):
    validate_distillation(lambda_preserve, temperature)
    if lambda_preserve <= 0:
        raise ValueError("v0.3 lambda-preserve must be positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha-replay must be finite and nonnegative")


def replay_preservation_loss(student_logits, teacher_logits, labels, image_mask,
                             temperature=1.0):
    """T² mean_R KL(Base_T || Student_T), with no CE or hard pseudo labels."""
    validate_distillation(1.0, temperature)
    expected = (labels.shape[0], 9, *labels.shape[1:])
    if student_logits.shape != expected or teacher_logits.shape != expected:
        raise ValueError("Expected matching NCHW replay logits with 9 classes")
    if image_mask.shape != labels.shape or image_mask.dtype != torch.bool:
        raise ValueError("image_mask must be boolean and match replay labels")
    if not torch.all(labels == IGNORE_INDEX):
        raise ValueError("Replay requires all-ignore labels; no CE is permitted")
    if not image_mask.any():
        raise ValueError("Replay has no real image pixels")
    student = student_logits.permute(0, 2, 3, 1)[image_mask]
    teacher = teacher_logits.detach().permute(0, 2, 3, 1)[image_mask]
    return temperature**2 * F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none", log_target=False,
    ).sum(dim=-1).mean()


def combine_losses(positive_loss, replay_kl, lambda_preserve=1.0, alpha=1.0):
    return positive_loss + alpha * lambda_preserve * replay_kl


def current_git_commit():
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else None
    except (OSError, subprocess.SubprocessError):
        return None


def state_digest(module, *, parameters_only=False):
    """Include BN buffers, using only one CPU tensor at a time for smoke checks."""
    digest = hashlib.sha256()
    values = module.named_parameters() if parameters_only else module.state_dict().items()
    for key, value in values:
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_model_invariants(student, teacher, optimizer, *, initial=False):
    student_ids = {id(p) for p in student.parameters()}
    teacher_ids = {id(p) for p in teacher.parameters()}
    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = {id(p) for p in student.parameters() if p.requires_grad}
    if student_ids & teacher_ids or optimized & teacher_ids or optimized != trainable:
        raise ValueError("Optimizer must contain only trainable student parameters, never teacher")
    if any(p.requires_grad for p in teacher.parameters()) or any(m.training for m in teacher.modules()):
        raise ValueError("Teacher must be frozen and eval")
    if any(p.requires_grad for p in student.encoder.parameters()) or any(m.training for m in student.encoder.modules()):
        raise ValueError("Encoder freeze invariant failed")
    expected = {id(p) for module in (student.decoder, student.segmentation_head) for p in module.parameters()}
    if trainable != expected:
        raise ValueError("Only decoder and segmentation head must be trainable")
    if initial:
        left, right = student.state_dict(), teacher.state_dict()
        if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
            raise ValueError("Student and teacher must start from identical original Base state")


def _metrics(positive, replay_sum, replay_pixels, replay_batches, weight, alpha):
    if not replay_pixels:
        raise ValueError("No replay pixels in epoch")
    replay_kl = replay_sum / replay_pixels
    total = combine_losses(positive["loss"], replay_kl, weight, alpha)
    return {
        **positive,
        "positive_ce_loss": positive["gsi_positive_ce_loss"],
        "unknown_preservation_loss": positive["base_preservation_kl_loss"],
        "replay_preservation_loss": replay_kl,
        "weighted_replay_preservation_loss": alpha * weight * replay_kl,
        "total_loss": total, "loss": total,
        "replay_preservation_pixel_count": replay_pixels,
        "replay_batch_count": replay_batches,
    }


def run_replay_epoch(model, loader, device, optimizer=None, *, teacher, replay_loader,
                     lambda_preserve=1.0, temperature=1.0, alpha=1.0,
                     verify_smoke=False):
    validate_replay(lambda_preserve, temperature, alpha)
    training = optimizer is not None
    model.train(training)
    model.encoder.eval()
    teacher.requires_grad_(False)
    teacher.eval()
    replay_sum, replay_pixels, replay_batches = 0.0, 0, 0

    def replay_forward(batch):
        images, labels, mask = (v.to(device) for v in batch)
        with torch.no_grad():
            targets = teacher(images)
        # alpha=0 also preserves v0.2 BN buffers and student RNG sequence.
        if training and alpha == 0:
            model.eval()
            with torch.no_grad():
                logits = model(images)
            model.train()
            model.encoder.eval()
        else:
            logits = model(images)
        kl = replay_preservation_loss(logits, targets, labels, mask, temperature)
        if verify_smoke and training and alpha > 0:
            gradient = torch.autograd.grad(kl, logits, retain_graph=True)[0].permute(0, 2, 3, 1)
            if torch.count_nonzero(gradient[~mask]):
                raise RuntimeError("Replay padding contributed to loss")
        return kl, int(mask.sum())

    with torch.set_grad_enabled(training):
        if not training:
            # Evaluate each validation example exactly once, including a longer
            # replay validation loader. No training replay is used here.
            positive = run_preservation_epoch(
                model, loader, device, teacher=teacher,
                lambda_preserve=lambda_preserve, temperature=temperature,
            )
            for batch in replay_loader:
                kl, count = replay_forward(batch)
                if not torch.isfinite(kl):
                    raise RuntimeError("Non-finite replay loss")
                replay_sum += kl.item() * count
                replay_pixels += count
                replay_batches += 1
        else:
            check_model_invariants(model, teacher, optimizer)
            if verify_smoke:
                before = [state_digest(m, parameters_only=i >= 2) for i, m in
                          enumerate((teacher, model.encoder, model.decoder, model.segmentation_head))]
            replay_iterator = iter(replay_loader)
            ce_sum = kl_sum = probability_sum = 0.0
            labeled = unknown_count = agreed = base_agreed = batches = 0
            for images, labels, mask in loader:
                try:
                    replay_batch = next(replay_iterator)
                except StopIteration:
                    replay_iterator = iter(replay_loader)  # No caching of image tensors.
                    try:
                        replay_batch = next(replay_iterator)
                    except StopIteration:
                        raise ValueError("Replay training loader is empty") from None
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    targets = teacher(images)
                logits = model(images)
                positive_loss, ce, unknown_kl = preservation_losses(
                    logits, targets, labels, mask, lambda_preserve, temperature,
                )
                replay_kl, count = replay_forward(replay_batch)
                total = combine_losses(positive_loss, replay_kl, lambda_preserve, alpha)
                if not all(torch.isfinite(v) for v in (positive_loss, ce, unknown_kl, replay_kl, total)):
                    raise RuntimeError("Non-finite v0.3 loss")
                total.backward()
                optimizer.step()
                with torch.no_grad():
                    p, u = mask & (labels == TARGET_CLASS), mask & (labels == IGNORE_INDEX)
                    p_count, u_count = int(p.sum()), int(u.sum())
                    ce_sum += ce.item() * p_count
                    kl_sum += unknown_kl.item() * u_count
                    labeled += p_count
                    unknown_count += u_count
                    prediction = logits.argmax(1)
                    agreed += int(((prediction == TARGET_CLASS) & p).sum())
                    base_agreed += int(((prediction == targets.argmax(1)) & u).sum())
                    probability_sum += logits.softmax(1)[:, TARGET_CLASS][p].double().sum().item()
                    replay_sum += replay_kl.item() * count
                    replay_pixels += count
                batches += 1
                replay_batches += 1
                if batches % 50 == 0:
                    print(f"  train: {batches} logical steps, CE={ce_sum/labeled:.6f}, "
                          f"replay={replay_sum/replay_pixels:.6f}", flush=True)
            if not labeled:
                raise ValueError("No labeled pixels in epoch")
            ce, kl = ce_sum / labeled, kl_sum / unknown_count if unknown_count else 0.0
            positive = {
                "loss": ce + lambda_preserve * kl,
                "gsi_positive_ce_loss": ce, "base_preservation_kl_loss": kl,
                "weighted_preservation_loss": lambda_preserve * kl,
                "labeled_pixel_count": labeled, "unknown_preservation_pixel_count": unknown_count,
                "class_7_labeled_pixel_agreement_recall": agreed / labeled,
                "class_7_mean_probability_on_labeled_pixels": probability_sum / labeled,
                "unknown_mean_kl_divergence": kl / temperature**2 if unknown_count else None,
                "unknown_student_base_argmax_agreement": base_agreed / unknown_count if unknown_count else None,
                "batch_count": batches,
            }
            if verify_smoke:
                after = [state_digest(m, parameters_only=i >= 2) for i, m in
                         enumerate((teacher, model.encoder, model.decoder, model.segmentation_head))]
                if before[:2] != after[:2] or any(a == b for a, b in zip(before[2:], after[2:])):
                    raise RuntimeError("Smoke invariant failed: teacher/encoder must stay fixed; decoder/head must change")
                if any(p.grad is not None for p in teacher.parameters()):
                    raise RuntimeError("Teacher received gradients")
                positive["smoke_checks"] = {
                    "teacher_unchanged": True, "encoder_unchanged": True,
                    "decoder_changed": True, "head_changed": True,
                    "replay_ce_computed": False,
                    "replay_padding_gradient_zero": True if alpha > 0 else None,
                }
    return _metrics(positive, replay_sum, replay_pixels, replay_batches, lambda_preserve, alpha)
