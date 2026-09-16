"""All-unknown Base-preservation v0.2; GSI positives may override the Base."""

import math

import torch
from torch.nn import functional as F

from src.model import build_model
from src.training.gsi_dataset import TARGET_CLASS
from src.training.prepare_gsi_labels import IGNORE_INDEX


LIMITATIONS = [
    "Preservation can retain errors made by the Base teacher.",
    "GSI unknown pixels can contain unlabeled Agriculture; preservation may inhibit adaptation there.",
    "SACLAJ is a point reference, not pixel-perfect ground truth.",
    "GSI latest imagery and SACLAJ observation dates may have temporal mismatch.",
    "lambda_preserve=1.0 and temperature=1.0 are Pilot conditions, not optimized values.",
    "v0.2 is an experimental Pilot, not a production model.",
]


def validate_distillation(lambda_preserve, temperature):
    if not math.isfinite(lambda_preserve) or lambda_preserve < 0:
        raise ValueError("lambda-preserve must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def build_teacher(base_path, device):
    # Model construction initializes CPU weights before loading the checkpoint.
    # Do not change the student's RNG sequence (in particular for lambda=0).
    with torch.random.fork_rng(devices=[]):
        teacher = build_model(base_path, device=device)
    teacher.requires_grad_(False)
    return teacher.eval()


def preservation_losses(student_logits, teacher_logits, labels, image_mask,
                        lambda_preserve=1.0, temperature=1.0):
    """CE mean over P + lambda * T² * mean_U KL(teacher_T || student_T).

    P = original-image pixels with label 7; U = original-image pixels with
    label 255. Sum KL over all nine classes before averaging unknown pixels.
    F.kl_div takes student log probabilities as input, teacher probabilities
    as target (log_target=False). Teacher targets are always detached.
    """
    validate_distillation(lambda_preserve, temperature)
    expected = (labels.shape[0], 9, *labels.shape[1:])
    if student_logits.shape != expected or teacher_logits.shape != expected:
        raise ValueError("Expected matching NCHW student/teacher logits with 9 classes")
    if image_mask.shape != labels.shape or image_mask.dtype != torch.bool:
        raise ValueError("image_mask must be a boolean mask matching labels")
    if not torch.all((labels == TARGET_CLASS) | (labels == IGNORE_INDEX)):
        raise ValueError("Only class-7/255 partial labels are supported")
    positive = image_mask & (labels == TARGET_CLASS)
    unknown = image_mask & (labels == IGNORE_INDEX)
    if not positive.any():
        raise ValueError("All-ignore batch has no training/validation signal")
    student = student_logits.permute(0, 2, 3, 1)
    teacher = teacher_logits.detach().permute(0, 2, 3, 1)
    positive_ce = F.cross_entropy(student[positive], labels[positive])
    if unknown.any():
        mean_kl = F.kl_div(
            F.log_softmax(student[unknown] / temperature, dim=-1),
            F.softmax(teacher[unknown] / temperature, dim=-1),
            reduction="none", log_target=False,
        ).sum(dim=-1).mean()
    else:
        # All-positive images are valid. No empty mean / NaN; no KL gradient.
        mean_kl = student.new_zeros(())
    preservation = temperature**2 * mean_kl
    return positive_ce + lambda_preserve * preservation, positive_ce, preservation


def run_preservation_epoch(model, loader, device, optimizer=None, *, teacher,
                           lambda_preserve=1.0, temperature=1.0):
    validate_distillation(lambda_preserve, temperature)
    training = optimizer is not None
    model.train(training)
    model.encoder.eval()
    teacher.requires_grad_(False)
    teacher.eval()  # Also fix all teacher BN buffers/dropout on every invocation.
    ce_sum, kl_sum, probability_sum = 0.0, 0.0, 0.0
    labeled, unknown_count, agreed, base_agreed = 0, 0, 0, 0
    batches = 0
    with torch.set_grad_enabled(training):
        for images, labels, image_mask in loader:
            images, labels, image_mask = images.to(device), labels.to(device), image_mask.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            logits = model(images)  # Exactly the same RGB tensor/preprocessing.
            loss, ce, kl = preservation_losses(
                logits, teacher_logits, labels, image_mask, lambda_preserve, temperature,
            )
            if not all(torch.isfinite(value) for value in (loss, ce, kl)):
                raise RuntimeError("Non-finite loss; aborting run")
            if training:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                positive = image_mask & (labels == TARGET_CLASS)
                unknown = image_mask & (labels == IGNORE_INDEX)
                count, u_count = int(positive.sum()), int(unknown.sum())
                ce_sum += ce.item() * count
                kl_sum += kl.item() * u_count
                labeled += count
                unknown_count += u_count
                prediction = logits.argmax(1)
                agreed += int(((prediction == TARGET_CLASS) & positive).sum())
                base_agreed += int(((prediction == teacher_logits.argmax(1)) & unknown).sum())
                probability_sum += logits.softmax(1)[:, TARGET_CLASS][positive].double().sum().item()
            batches += 1
            if batches % 50 == 0:
                phase = "train" if training else "validation"
                print(f"  {phase}: {batches} batches, CE={ce_sum / labeled:.6f}, "
                      f"preservation={kl_sum / unknown_count if unknown_count else 0:.6f}", flush=True)
    if labeled == 0:
        raise ValueError("No labeled pixels in epoch")
    ce = ce_sum / labeled
    kl = kl_sum / unknown_count if unknown_count else 0.0
    return {
        "loss": ce + lambda_preserve * kl,
        "gsi_positive_ce_loss": ce,
        "base_preservation_kl_loss": kl,
        "weighted_preservation_loss": lambda_preserve * kl,
        "labeled_pixel_count": labeled,
        "unknown_preservation_pixel_count": unknown_count,
        "class_7_labeled_pixel_agreement_recall": agreed / labeled,
        "class_7_mean_probability_on_labeled_pixels": probability_sum / labeled,
        "unknown_mean_kl_divergence": kl / temperature**2 if unknown_count else None,
        "unknown_student_base_argmax_agreement": base_agreed / unknown_count if unknown_count else None,
        "batch_count": batches,
    }


def provenance(base_sha, lambda_preserve, temperature):
    return {
        "training_mode": "base_preservation",
        "teacher_checkpoint_sha256": base_sha,
        "student_initialization_checkpoint_sha256": base_sha,
        "teacher_equals_student_initialization_base": True,
        "teacher": {"frozen": True, "eval": True, "no_grad": True,
                    "optimizer_included": False, "batchnorm_statistics_updated": False},
        "loss": "positive_CE + lambda_preserve * preservation_KL",
        "distillation_loss_definition": "T^2 * mean_unknown(sum_c p_base_T[c] * (log(p_base_T[c]) - log(p_student_T[c])))",
        "kl_direction": "KL(Base teacher || student)",
        "pytorch_kl_inputs": "input=log_softmax(student_logits/T), target=softmax(detached_teacher_logits/T), reduction=none, log_target=False; sum classes",
        "temperature": temperature,
        "lambda_preserve": lambda_preserve,
        "positive_loss_normalization": "mean over original-image positive pixels, independently within each batch",
        "preservation_loss_normalization": "sum over 9 classes, mean over original-image unknown pixels, independently within each batch; multiply by T^2",
        "positive_mask_definition": "image_mask AND label == 7",
        "preservation_mask_definition": "image_mask AND label == 255; no positive or synthetic padding pixels",
        "epoch_loss_aggregation": "CE weighted by positive count; preservation weighted by unknown count; loss = epoch_CE + lambda_preserve * epoch_preservation",
        "best_criterion": "minimum validation combined loss; first epoch wins ties",
    }
