"""Synthetic-only v0.2 tests; no real checkpoint, GSI or SACLAJ data."""

import copy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src import model as shared_model
from src.training import base_preservation as preservation
from src.training import train_gsi_paddy as training
from src.training.gsi_dataset import (
    GsiPaddyDataset, collate_padded, collate_preservation, scan_dataset,
)
from test_train_gsi_paddy import TinyModel, make_data, single_thread  # noqa: F401


def example():
    generator = torch.Generator().manual_seed(19)
    student = torch.randn(1, 9, 2, 3, generator=generator, requires_grad=True)
    teacher = torch.randn(1, 9, 2, 3, generator=generator, requires_grad=True)
    labels = torch.tensor([[[7, 255, 255], [7, 255, 255]]])
    extent = torch.tensor([[[True, True, True], [True, True, False]]])
    return student, teacher, labels, extent


def test_t1_direction_and_independent_pixel_means():
    student, teacher, labels, extent = example()
    total, ce, kl = preservation.preservation_losses(student, teacher, labels, extent)
    positive, unknown = extent & (labels == 7), extent & (labels == 255)
    s = student.permute(0, 2, 3, 1)
    t = teacher.permute(0, 2, 3, 1)
    expected_ce = -s.log_softmax(-1)[positive][:, 7].mean()
    p, log_p, log_q = t[unknown].softmax(-1), t[unknown].log_softmax(-1), s[unknown].log_softmax(-1)
    expected_kl = (p * (log_p - log_q)).sum(-1).mean()
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(total, expected_ce + expected_kl)
    reverse_kl = (log_q.exp() * (log_q - log_p)).sum(-1).mean()
    assert not torch.isclose(kl, reverse_kl)
    # Multiplying the number of unknowns must not change either component mean.
    positions = [0, 3, 1, 2, 4, 1, 2, 4]
    repeated = preservation.preservation_losses(
        student.reshape(1, 9, 1, 6)[..., positions],
        teacher.reshape(1, 9, 1, 6)[..., positions],
        labels.reshape(1, 1, 6)[..., positions],
        extent.reshape(1, 1, 6)[..., positions],
    )
    for actual, expected in zip(repeated, (total, ce, kl)):
        torch.testing.assert_close(actual, expected)


def test_gradient_masks_teacher_detach_and_padding_exclusion():
    student, teacher, labels, extent = example()
    _, ce, kl = preservation.preservation_losses(student, teacher, labels, extent)
    ce_gradient = torch.autograd.grad(ce, student, retain_graph=True)[0].permute(0, 2, 3, 1)
    kl_gradient = torch.autograd.grad(kl, student, retain_graph=True)[0].permute(0, 2, 3, 1)
    positive, unknown = extent & (labels == 7), extent & (labels == 255)
    assert torch.count_nonzero(ce_gradient[~positive]) == 0
    assert torch.count_nonzero(ce_gradient[positive]) > 0
    assert torch.count_nonzero(kl_gradient[~unknown]) == 0
    assert torch.count_nonzero(kl_gradient[unknown]) > 0
    (ce + kl).backward()
    assert teacher.grad is None
    modified = teacher.detach().clone().permute(0, 2, 3, 1)
    modified[~unknown] = 100.0
    changed = preservation.preservation_losses(student, modified.permute(0, 3, 1, 2), labels, extent)
    torch.testing.assert_close(changed[2], kl)


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_identical_distribution_zero_and_departure_positive(temperature):
    student, _, labels, extent = example()
    _, _, zero = preservation.preservation_losses(student, student, labels, extent, temperature=temperature)
    assert abs(zero.item()) < 1e-6
    shifted = student.detach().clone()
    shifted[:, 7] += 5
    _, _, kl = preservation.preservation_losses(shifted, student, labels, extent, temperature=temperature)
    assert kl.item() > 0
    unknown = extent & (labels == 255)
    p = (student.permute(0, 2, 3, 1)[unknown] / temperature).softmax(-1)
    q = (shifted.permute(0, 2, 3, 1)[unknown] / temperature).softmax(-1)
    torch.testing.assert_close(kl, temperature**2 * (p * (p.log() - q.log())).sum(-1).mean())


def test_lambda_zero_has_exactly_ce_value_and_gradient():
    student, teacher, labels, extent = example()
    total, ce, kl = preservation.preservation_losses(student, teacher, labels, extent, lambda_preserve=0)
    assert kl > 0
    torch.testing.assert_close(total, ce, rtol=0, atol=0)
    total_gradient = torch.autograd.grad(total, student, retain_graph=True)[0]
    ce_gradient = torch.autograd.grad(ce, student)[0]
    torch.testing.assert_close(total_gradient, ce_gradient, rtol=0, atol=0)


def test_empty_unknown_and_all_ignore():
    student, teacher, labels, extent = example()
    total, ce, kl = preservation.preservation_losses(student, teacher, torch.full_like(labels, 7), extent)
    assert kl.item() == 0 and torch.isfinite(total)
    torch.testing.assert_close(total, ce)
    with pytest.raises(ValueError, match="All-ignore"):
        preservation.preservation_losses(student, teacher, torch.full_like(labels, 255), extent)
    with pytest.raises(ValueError, match="class-7/255"):
        preservation.preservation_losses(student, teacher, torch.zeros_like(labels), extent)


@pytest.mark.parametrize("weight,temperature", [(-1, 1), (float("nan"), 1), (float("inf"), 1),
                                              (1, 0), (1, -1), (1, float("nan")), (1, float("inf"))])
def test_invalid_hyperparameters(weight, temperature):
    with pytest.raises(ValueError):
        preservation.preservation_losses(*example(), lambda_preserve=weight, temperature=temperature)


def test_padding_same_rgb_labels_but_preservation_only_inside_images():
    batch = [(torch.rand(3, 572, 572), torch.full((572, 572), 255)),
             (torch.rand(3, 21, 35), torch.full((21, 35), 7))]
    images, labels, extent = collate_preservation(batch)
    old_images, old_labels = collate_padded(batch)
    torch.testing.assert_close(images, old_images)
    torch.testing.assert_close(labels, old_labels)
    assert images.shape == (2, 3, 576, 576)
    assert int((extent & (labels == 255)).sum()) == 572 * 572
    assert int((extent & (labels == 7)).sum()) == 21 * 35
    assert not extent[:, 572:].any() and not extent[:, :, 572:].any()


def test_teacher_lifecycle_optimizer_and_lambda_zero_v01_equivalence(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kwargs: TinyModel())
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    student = shared_model.build_model(base)
    training.freeze_encoder(student)
    rng_before = torch.get_rng_state().clone()
    teacher = preservation.build_teacher(base, "cpu")
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(student.parameters(), teacher.parameters()))
    optimizer = training.make_optimizer(student, 1e-4)
    teacher_ids = {id(p) for p in teacher.parameters()}
    assert not teacher_ids & {id(p) for g in optimizer.param_groups for p in g["params"]}
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    seen = []
    teacher.register_forward_pre_hook(lambda module, inputs: seen.append(
        (module.training, torch.is_grad_enabled(), inputs[0].data_ptr())))
    student_inputs = []
    student.register_forward_pre_hook(lambda module, inputs: student_inputs.append(inputs[0].data_ptr()))
    org, prepared = make_data(tmp_path)
    samples, _ = scan_dataset(org, prepared / "labels")
    dataset = GsiPaddyDataset(samples)
    legacy = copy.deepcopy(student)
    old_metrics = training.run_epoch(legacy, DataLoader(dataset, collate_fn=collate_padded), "cpu",
                                     training.make_optimizer(legacy, 1e-4))
    student_inputs.clear()
    loader = DataLoader(dataset, collate_fn=collate_preservation)
    teacher.train()  # Runner restores eval even after an accidental train() call.
    metrics = preservation.run_preservation_epoch(student, loader, "cpu", optimizer,
                                                  teacher=teacher, lambda_preserve=0)
    assert metrics["weighted_preservation_loss"] == 0
    assert metrics["loss"] == pytest.approx(old_metrics["loss"], abs=1e-6)
    for k, value in student.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[k], rtol=1e-5, atol=1e-6)
    assert [item[2] for item in seen] == student_inputs
    assert all(not is_training and not grad for is_training, grad, _ in seen)
    preservation.run_preservation_epoch(student, loader, "cpu", teacher=teacher)
    assert all(not module.training for module in teacher.modules())
    assert all(p.grad is None for p in teacher.parameters())
    for k, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)


def test_epoch_aggregation_uses_separate_denominators_and_combined_validation():
    class PixelModel(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.encoder, self.scale = nn.Identity(), scale

        def forward(self, images):
            logits = images.new_zeros((images.shape[0], 9, *images.shape[2:]))
            logits[:, 7] = images[:, 0] * self.scale
            return logits

    student, teacher = PixelModel(2), PixelModel(1)
    batches = [(torch.ones(1, 3, 2, 2), torch.tensor([[[7, 255], [255, 255]]]), torch.ones(1, 2, 2, dtype=torch.bool)),
               (torch.full((1, 3, 2, 2), 3.0), torch.tensor([[[7, 7], [7, 255]]]), torch.ones(1, 2, 2, dtype=torch.bool))]
    components = [preservation.preservation_losses(student(x), teacher(x), y, m, 0.6, 2.0)
                  for x, y, m in batches]
    metrics = preservation.run_preservation_epoch(student, batches, "cpu", teacher=teacher,
                                                  lambda_preserve=0.6, temperature=2.0)
    ce = (components[0][1].item() + 3 * components[1][1].item()) / 4
    kl = (3 * components[0][2].item() + components[1][2].item()) / 4
    assert metrics["gsi_positive_ce_loss"] == pytest.approx(ce)
    assert metrics["base_preservation_kl_loss"] == pytest.approx(kl)
    assert metrics["loss"] == pytest.approx(ce + 0.6 * kl)
    assert metrics["unknown_mean_kl_divergence"] == pytest.approx(kl / 4)
    assert metrics["unknown_student_base_argmax_agreement"] == 1
    assert metrics["labeled_pixel_count"] == metrics["unknown_preservation_pixel_count"] == 4
    empty_unknown = [(batches[0][0], torch.full((1, 2, 2), 7), batches[0][2])]
    metrics = preservation.run_preservation_epoch(student, empty_unknown, "cpu", teacher=teacher)
    assert metrics["base_preservation_kl_loss"] == 0
    assert metrics["unknown_mean_kl_divergence"] is None
    assert metrics["unknown_student_base_argmax_agreement"] is None


@pytest.mark.parametrize("purpose", ["training", "preflight", "smoke_test"])
def test_synthetic_run_provenance_steps_and_strict_checkpoint_load(tmp_path, monkeypatch, purpose):
    import segmentation_models_pytorch as smp
    built = []
    def factory(**kwargs):
        model = TinyModel()
        built.append(model)
        return model
    monkeypatch.setattr(smp, "Unet", factory)
    org, prepared = make_data(tmp_path)
    base = tmp_path / "base.pth"
    state = TinyModel().state_dict()
    torch.save(state, base)
    options = [] if purpose == "training" else ["--" + purpose.replace("_", "-")]
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared),
        "--base-model", str(base), "--base-sha256", training._sha256(base),
        "--training-mode", "base_preservation", "--output-dir", str(tmp_path / "runs"),
        "--num-threads", "1", *options,
    ])
    steps = []
    original_step = torch.optim.AdamW.step
    def step(self, *args, **kwargs):
        steps.append(1)
        return original_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    run = training.train(args)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert len(built) == 2
    student, teacher = built
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(not module.training for module in teacher.modules())
    assert manifest["schema_version"] == 2
    assert manifest["epochs"] == 1
    assert manifest["run_purpose"] == purpose
    assert manifest["training_mode"] == "base_preservation"
    assert manifest["teacher_checkpoint_sha256"] == manifest["student_initialization_checkpoint_sha256"] == training._sha256(base)
    assert manifest["teacher_equals_student_initialization_base"] is True
    assert manifest["teacher"] == {"frozen": True, "eval": True, "no_grad": True,
                                   "optimizer_included": False, "batchnorm_statistics_updated": False}
    for key in ("distillation_loss_definition", "kl_direction", "temperature", "lambda_preserve",
                "positive_loss_normalization", "preservation_loss_normalization",
                "positive_mask_definition", "preservation_mask_definition", "epoch_loss_aggregation"):
        assert key in manifest
    assert manifest["temperature"] == manifest["lambda_preserve"] == 1
    assert set(preservation.LIMITATIONS) <= set(manifest["known_limitations"])
    assert manifest["train_count"] == 4 and manifest["validation_count"] == 1
    assert manifest["excluded_all_ignore_images"] == 1
    if purpose == "preflight":
        assert not steps
        assert manifest["status"] == "preflight_passed"
        assert manifest["epoch_metrics"] == []
        assert manifest["final_checkpoint"] is None
        assert not list((run / "checkpoints").iterdir())
        assert len(manifest["preflight_metrics"]) == 2
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        return
    assert len(steps) == (1 if purpose == "smoke_test" else 4)
    assert manifest["status"] == ("smoke_test_completed" if purpose == "smoke_test" else "completed")
    assert len(manifest["epoch_metrics"]) == 1
    for split in ("training", "validation"):
        metrics = manifest["epoch_metrics"][0][split]
        assert metrics["loss"] == pytest.approx(metrics["gsi_positive_ce_loss"] + metrics["base_preservation_kl_loss"])
        assert metrics["labeled_pixel_count"] > 0 and metrics["unknown_preservation_pixel_count"] > 0
    assert manifest["best_validation_loss"] == manifest["epoch_metrics"][0]["validation"]["loss"]
    for filename in ("epoch_001.pth", "best.pth", "final.pth"):
        checkpoint = run / "checkpoints" / filename
        saved = torch.load(checkpoint, weights_only=True)
        assert all(v.device.type == "cpu" for v in saved.values())
        loaded = shared_model.build_model(checkpoint)
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, student.state_dict()[key], rtol=0, atol=0)
        assert loaded(torch.zeros(1, 3, 32, 32)).shape == (1, 9, 32, 32)
    if purpose == "training":
        again = training.train(args)
        repeat = json.loads((again / "run_manifest.json").read_text())
        assert repeat["epoch_metrics"] == manifest["epoch_metrics"]
        assert (again / "train_ids.json").read_bytes() == (run / "train_ids.json").read_bytes()


def test_expected_base_hash_rejected_before_training(tmp_path):
    base = tmp_path / "base.pth"
    base.write_bytes(b"synthetic checkpoint")
    args = training.make_parser().parse_args([
        "--org-dir", str(tmp_path), "--prepared-dir", str(tmp_path),
        "--base-model", str(base), "--base-sha256", "0" * 64,
        "--training-mode", "base_preservation",
    ])
    with pytest.raises(ValueError, match="SHA256"):
        training.train(args)


def test_smoke_rejects_multiple_epochs():
    args = training.make_parser().parse_args([
        "--org-dir", ".", "--prepared-dir", ".", "--training-mode", "base_preservation",
        "--smoke-test", "--epochs", "3",
    ])
    with pytest.raises(ValueError, match="one optimizer step"):
        training.train(args)


def test_default_v01_keeps_three_epochs_without_teacher(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kwargs: TinyModel())
    def forbidden(*args, **kwargs):
        raise AssertionError("v0.1 must not build a teacher")
    monkeypatch.setattr(training, "build_teacher", forbidden)
    org, prepared = make_data(tmp_path)
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared), "--base-model", str(base),
        "--output-dir", str(tmp_path / "runs"), "--num-threads", "1",
    ])
    run = training.train(args)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["training_mode"] == "positive_only"
    assert manifest["epochs"] == len(manifest["epoch_metrics"]) == 3
    assert "teacher_checkpoint_sha256" not in manifest


def test_checkpoint_loader_rejects_missing_key(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kwargs: TinyModel())
    checkpoint = tmp_path / "student.pth"
    training.save_checkpoint(checkpoint, TinyModel())
    state = torch.load(checkpoint, weights_only=True)
    del state["segmentation_head.weight"]
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        shared_model.build_model(checkpoint)
