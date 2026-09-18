"""Synthetic-only v0.3 regression tests; no GSI/SACLAJ data or Base required."""

import copy
from dataclasses import replace
import json
import random
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training import base_preservation as v02
from src.training import replay_preservation as replay
from src.training import train_gsi_paddy as training
from src.training.gsi_dataset import (
    GsiPaddyDataset, Sample, collate_preservation, scan_dataset, split_samples,
)
from src.training.prepare_gsi_labels import prepare_gsi_dataset
from test_base_preservation import example
from test_train_gsi_paddy import TinyModel, make_data, single_thread  # noqa: F401


def make_replay_data(tmp_path):
    org, prepared = make_data(tmp_path)
    # Add three all-ignore images to the one provided by the v0.2 fixture.
    for index in range(6, 9):
        for folder in ("org", "val"):
            Image.new("RGB", (5, 3)).save(org.parent / folder / "nested" / f"{index}.png")
    prepare_gsi_dataset(org.parent, prepared, gsi_category="paddy", oem_class_id=7,
                        label_color=(0, 255, 255))
    return org, prepared


def test_scan_dataset_separates_positive_replay_and_rejects_changes(tmp_path):
    org, prepared = make_data(tmp_path)
    positive, ignored = scan_dataset(org, prepared / "labels")
    assert len(positive) == 5 and len(ignored) == 1
    assert not {s.source_image_id for s in positive} & {s.source_image_id for s in ignored}
    with pytest.raises(ValueError, match="All-ignore"):
        GsiPaddyDataset(ignored)[0]
    with pytest.raises(ValueError, match="all-ignore"):
        replay.GsiReplayDataset(positive)
    dataset = replay.GsiReplayDataset(ignored)
    image, label = dataset[0]
    assert image.shape == (3, 3, 5) and torch.all(label == 255)
    changed = np.full((3, 5), 255, dtype=np.uint8)
    changed[0, 0] = 7
    Image.fromarray(changed).save(ignored[0].label_path)
    with pytest.raises(ValueError, match="all-ignore"):
        dataset[0]


def test_positive_ids_exactly_match_v02_split_and_replay_is_deterministic(tmp_path):
    def samples(prefix, count, positives):
        return [Sample(f"{prefix}/{i:04}", tmp_path, tmp_path, 1, 1, positives, "", "") for i in range(count)]
    positives, ignored = samples("p", 1286, 1), samples("r", 1314, 0)
    before = random.getstate()
    train, validation = split_samples(positives, 0.8, 42)
    v03_train, v03_validation = split_samples(positives, 0.8, 42)
    r_train, r_validation = replay.replay_splits(v03_train, v03_validation, ignored)
    assert (v03_train, v03_validation) == (train, validation)
    assert (len(train), len(validation)) == (1028, 258)
    assert (len(r_train), len(r_validation)) == (1051, 263)
    assert (r_train, r_validation) == replay.replay_splits(train, validation, list(reversed(ignored)))
    assert random.getstate() == before
    groups = [{s.source_image_id for s in group} for group in (train, validation, r_train, r_validation)]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 2600
    for a, b in ((train, validation + [train[0]]), (train, validation)):
        conflicting = ignored if len(b) > len(validation) else [replace(s, source_image_id=train[0].source_image_id) for s in ignored[:2]]
        with pytest.raises(ValueError, match="Duplicate|disjoint"):
            replay.replay_splits(a, b, conflicting)


def test_replay_padding_has_original_extent_and_same_v02_preprocessing():
    batch = [(torch.rand(3, 572, 572), torch.full((572, 572), 255)),
             (torch.rand(3, 19, 35), torch.full((19, 35), 255))]
    images, labels, mask = replay.checked_collate(batch)
    assert images.shape == (2, 3, 576, 576)
    assert int(mask.sum()) == 572 * 572 + 19 * 35
    assert not mask[0, 572:].any() and not mask[0, :, 572:].any()
    assert not mask[1, 19:].any() and not mask[1, :, 35:].any()
    assert torch.all(labels == 255)
    torch.testing.assert_close(images[0, :, :572, :572], batch[0][0])


def test_checked_collate_rejects_padding_mask_regression(monkeypatch):
    def broken(batch):
        images, labels, mask = collate_preservation(batch)
        return images, labels, torch.ones_like(mask)
    monkeypatch.setattr(replay, "collate_preservation", broken)
    with pytest.raises(ValueError, match="Padding mask"):
        replay.checked_collate([(torch.zeros(3, 3, 5), torch.full((3, 5), 255))])


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_replay_real_pixel_mean_kl_direction_no_ce_and_padding_gradient(temperature, monkeypatch):
    student, teacher, labels, mask = example()
    labels.fill_(255)
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not compute CE")
    monkeypatch.setattr(replay.F, "cross_entropy", forbidden)
    kl = replay.replay_preservation_loss(student, teacher, labels, mask, temperature)
    s, t = student.permute(0, 2, 3, 1)[mask], teacher.detach().permute(0, 2, 3, 1)[mask]
    p, log_p, log_q = (t / temperature).softmax(-1), (t / temperature).log_softmax(-1), (s / temperature).log_softmax(-1)
    torch.testing.assert_close(kl, temperature**2 * (p * (log_p - log_q)).sum(-1).mean())
    kl.backward()
    assert torch.count_nonzero(student.grad.permute(0, 2, 3, 1)[~mask]) == 0
    assert torch.count_nonzero(student.grad.permute(0, 2, 3, 1)[mask]) > 0
    assert teacher.grad is None
    modified = student.detach().clone().permute(0, 2, 3, 1)
    modified[~mask] = float("nan")
    changed = replay.replay_preservation_loss(modified.permute(0, 3, 1, 2), teacher, labels, mask, temperature)
    torch.testing.assert_close(changed, kl)
    repeated = replay.replay_preservation_loss(student.repeat(1, 1, 2, 1), teacher.repeat(1, 1, 2, 1),
                                               labels.repeat(1, 2, 1), mask.repeat(1, 2, 1), temperature)
    torch.testing.assert_close(repeated, kl)


def test_alpha_zero_loss_and_gradient_equal_v02():
    student, teacher, labels, mask = example()
    positive, _, _ = v02.preservation_losses(student, teacher, labels, mask)
    r_student = student.detach().clone().requires_grad_()
    kl = replay.replay_preservation_loss(r_student, teacher, torch.full_like(labels, 255), mask)
    total = replay.combine_losses(positive, kl, alpha=0)
    torch.testing.assert_close(total, positive, rtol=0, atol=0)
    a = torch.autograd.grad(positive, student, retain_graph=True)[0]
    b, r = torch.autograd.grad(total, (student, r_student))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.count_nonzero(r) == 0


@pytest.mark.parametrize("weight,temperature,alpha", [(0, 1, 1), (-1, 1, 1), (1, 0, 1),
                                                      (1, 1, -1), (1, 1, float("nan")), (1, 1, float("inf"))])
def test_invalid_replay_hyperparameters(weight, temperature, alpha):
    with pytest.raises(ValueError):
        replay.validate_replay(weight, temperature, alpha)


def test_replay_rejects_positive_and_empty_masks():
    student, teacher, labels, mask = example()
    with pytest.raises(ValueError, match="all-ignore"):
        replay.replay_preservation_loss(student, teacher, labels, mask)
    with pytest.raises(ValueError, match="no real"):
        replay.replay_preservation_loss(student, teacher, torch.full_like(labels, 255), torch.zeros_like(mask))


def batches():
    images = torch.rand(1, 3, 3, 5)
    labels = torch.full((1, 3, 5), 255)
    labels[:, 1, 1] = 7
    mask = torch.ones_like(labels, dtype=torch.bool)
    return (images, labels, mask), (images.clone(), torch.full_like(labels, 255), mask.clone())


@pytest.mark.parametrize("replay_length", [1, 5])
def test_one_step_per_positive_batch_cycles_short_replay_and_full_validation(replay_length):
    student = TinyModel()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    training.freeze_encoder(student)
    optimizer = training.make_optimizer(student, 1e-4)
    p, r = batches()
    before = replay.state_digest(teacher), replay.state_digest(student.encoder)
    steps = []
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    seen = []
    teacher.register_forward_pre_hook(lambda m, i: seen.append((m.training, torch.is_grad_enabled())))
    metrics = replay.run_replay_epoch(student, [p] * 3, "cpu", optimizer, teacher=teacher,
                                      replay_loader=[r] * replay_length)
    assert len(steps) == metrics["batch_count"] == metrics["replay_batch_count"] == 3
    assert before == (replay.state_digest(teacher), replay.state_digest(student.encoder))
    assert all(not a and not b for a, b in seen)
    assert all(p.grad is None for p in teacher.parameters())
    assert not {id(p) for p in teacher.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]}
    result = replay.run_replay_epoch(student, [p] * 3, "cpu", teacher=teacher, replay_loader=[r] * replay_length)
    assert result["batch_count"] == 3 and result["replay_batch_count"] == replay_length
    assert result["replay_preservation_pixel_count"] == 15 * replay_length
    assert result["total_loss"] == pytest.approx(result["positive_ce_loss"] + result["unknown_preservation_loss"] + result["replay_preservation_loss"])


def test_alpha_zero_preserves_v02_updates_even_with_decoder_bn_and_dropout():
    student = TinyModel()
    student.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.Dropout(0.2), nn.ReLU())
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    training.freeze_encoder(student)
    old = copy.deepcopy(student)
    p, r = batches()
    torch.manual_seed(47)
    v02.run_preservation_epoch(old, [p] * 3, "cpu", training.make_optimizer(old, 1e-4), teacher=teacher)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(47)
    replay.run_replay_epoch(student, [p] * 3, "cpu", training.make_optimizer(student, 1e-4),
                            teacher=teacher, replay_loader=[r], alpha=0)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, student.state_dict()[key], rtol=0, atol=0)


def test_validation_uses_independent_pixel_weighting_for_p_u_r():
    class PixelModel(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.encoder, self.scale = nn.Identity(), scale

        def forward(self, images):
            logits = images.new_zeros((images.shape[0], 9, *images.shape[2:]))
            logits[:, 7] = images[:, 0] * self.scale
            return logits

    student, teacher = PixelModel(2), PixelModel(1)
    positive = [
        (torch.ones(1, 3, 2, 2), torch.tensor([[[7, 255], [255, 255]]]), torch.ones(1, 2, 2, dtype=torch.bool)),
        (torch.full((1, 3, 2, 2), 3.0), torch.tensor([[[7, 7], [7, 255]]]), torch.ones(1, 2, 2, dtype=torch.bool)),
    ]
    ignored = [(x, torch.full_like(y, 255), m.clone()) for x, y, m in positive]
    ignored[0][2][:, 1, :] = False
    weight, temperature, alpha = 0.6, 2.0, 1.7
    components = [v02.preservation_losses(student(x), teacher(x), y, m, weight, temperature) for x, y, m in positive]
    replay_kl = [replay.replay_preservation_loss(student(x), teacher(x), y, m, temperature).item() for x, y, m in ignored]
    ce = (components[0][1].item() + 3 * components[1][1].item()) / 4
    unknown = (3 * components[0][2].item() + components[1][2].item()) / 4
    r = (2 * replay_kl[0] + 4 * replay_kl[1]) / 6
    metrics = replay.run_replay_epoch(student, positive, "cpu", teacher=teacher, replay_loader=ignored,
                                      lambda_preserve=weight, temperature=temperature, alpha=alpha)
    assert metrics["positive_ce_loss"] == pytest.approx(ce)
    assert metrics["unknown_preservation_loss"] == pytest.approx(unknown)
    assert metrics["replay_preservation_loss"] == pytest.approx(r)
    assert metrics["total_loss"] == pytest.approx(ce + weight * unknown + alpha * weight * r)


def test_smoke_detects_missing_optimizer_update():
    student = TinyModel()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    training.freeze_encoder(student)
    optimizer = training.make_optimizer(student, 1e-4)
    optimizer.step = lambda: None
    p, r = batches()
    with pytest.raises(RuntimeError, match="Smoke invariant"):
        replay.run_replay_epoch(student, [p], "cpu", optimizer, teacher=teacher,
                                replay_loader=[r], verify_smoke=True)


@pytest.mark.parametrize("purpose", ["preflight", "smoke-test", "training"])
def test_cli_v03_manifest_splits_smoke_and_steps(tmp_path, monkeypatch, purpose):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kw: TinyModel())
    monkeypatch.setattr(replay, "PILOT_COUNTS", (5, 4))
    org, prepared = make_replay_data(tmp_path)
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    options = [] if purpose == "training" else ["--" + purpose]
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared), "--base-model", str(base),
        "--base-sha256", training._sha256(base), "--training-mode", "base_preservation_replay",
        "--output-dir", str(tmp_path / "runs"), "--num-threads", "1", *options,
    ])
    run = training.train(args)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["training_mode"] == "base_preservation_replay"
    assert manifest["model_version"] == "gsi_phase_a_v0.3" and manifest["schema_version"] == 3
    assert manifest["lambda_preserve"] == manifest["temperature"] == manifest["alpha_replay"] == 1
    assert manifest["epochs"] == 1
    assert manifest["positive_train_count"] == 4 and manifest["positive_validation_count"] == 1
    assert manifest["replay_train_count"] == 3 and manifest["replay_validation_count"] == 1
    assert manifest["teacher_checkpoint_sha256"] == manifest["base_model_sha256"] == training._sha256(base)
    assert set(manifest["versions"]) == {"python", "torch", "segmentation_models_pytorch", "numpy", "Pillow"}
    assert "git_commit_sha" in manifest
    for prefix, file in (("positive_train", "train_ids.json"), ("positive_validation", "validation_ids.json"),
                         ("replay_train", "replay_train_ids.json"), ("replay_validation", "replay_validation_ids.json")):
        assert manifest[prefix + "_ids_sha256"] == training._sha256(run / file)
    positive, ignored = scan_dataset(org, prepared / "labels")
    a, b = split_samples(positive, 0.8, 42)
    assert json.loads((run / "train_ids.json").read_text()) == [s.source_image_id for s in a]
    assert json.loads((run / "validation_ids.json").read_text()) == [s.source_image_id for s in b]
    if purpose == "preflight":
        assert manifest["status"] == "preflight_passed" and manifest["best_checkpoint_sha256"] is None
        assert not list((run / "checkpoints").iterdir())
        metrics = manifest["preflight_metrics"]
    else:
        assert manifest["status"] == ("completed" if purpose == "training" else "smoke_test_completed")
        assert manifest["best_checkpoint_sha256"] == training._sha256(run / "checkpoints/best.pth")
        metrics = manifest["epoch_metrics"][0]
        assert metrics["training"]["batch_count"] == (4 if purpose == "training" else 1)
        if purpose == "smoke-test":
            checks = metrics["training"]["smoke_checks"]
            assert checks["replay_ce_computed"] is False
            assert all(v for k, v in checks.items() if k != "replay_ce_computed")
        else:
            again = training.train(args)
            assert json.loads((again / "run_manifest.json").read_text())["epoch_metrics"] == manifest["epoch_metrics"]
    for field in (("train_sample", "validation_sample") if purpose == "preflight" else ("training", "validation")):
        values = metrics[field]
        assert values["labeled_pixel_count"] > 0 and values["unknown_preservation_pixel_count"] > 0
        assert values["replay_preservation_pixel_count"] > 0
        assert values["total_loss"] == pytest.approx(sum(values[k] for k in ("positive_ce_loss", "unknown_preservation_loss", "replay_preservation_loss")))


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.CalledProcessError(1, "git"), subprocess.TimeoutExpired("git", 5)])
def test_git_unavailable_records_null(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(replay.subprocess, "run", fail)
    assert replay.current_git_commit() is None


def test_git_commit_uses_current_repository(monkeypatch):
    def run(args, **kwargs):
        assert args == ["git", "rev-parse", "HEAD"]
        assert (kwargs["cwd"] / "src/training/train_gsi_paddy.py").is_file()
        return subprocess.CompletedProcess(args, 0, "a" * 40 + "\n")
    monkeypatch.setattr(replay.subprocess, "run", run)
    assert replay.current_git_commit() == "a" * 40


def test_preflight_rejects_unexpected_counts_before_model_load(tmp_path):
    org, prepared = make_data(tmp_path)
    base = tmp_path / "base.pth"
    base.write_bytes(b"unused")
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared), "--base-model", str(base),
        "--training-mode", "base_preservation_replay", "--preflight",
    ])
    with pytest.raises(ValueError, match="1286, 1314"):
        training.train(args)


def test_preflight_rejects_identity_and_optimizer_violations():
    student, teacher = TinyModel(), TinyModel().requires_grad_(False).eval()
    training.freeze_encoder(student)
    optimizer = training.make_optimizer(student, 1e-4)
    with pytest.raises(ValueError, match="identical"):
        replay.check_model_invariants(student, teacher, optimizer, initial=True)
    optimizer.add_param_group({"params": list(teacher.parameters())})
    with pytest.raises(ValueError, match="Optimizer"):
        replay.check_model_invariants(student, teacher, optimizer)


@pytest.mark.parametrize("flag,value", [("--seed", "43"), ("--train-ratio", "0.7")])
def test_v03_rejects_changed_positive_split(flag, value):
    args = training.make_parser().parse_args([
        "--org-dir", ".", "--prepared-dir", ".", "--training-mode", "base_preservation_replay", flag, value,
    ])
    with pytest.raises(ValueError, match="v0.2 Pilot positive split"):
        training.train(args)
