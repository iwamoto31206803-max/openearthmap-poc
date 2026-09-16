import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import predict_geotiff_tiled as inference
from src.training import train_gsi_paddy as training
from src.training.gsi_dataset import (
    GsiPaddyDataset, collate_padded, scan_dataset, split_samples,
)
from src.training.prepare_gsi_labels import prepare_gsi_dataset


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout2d(0.5))
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.segmentation_head = nn.Conv2d(4, 9, 1)

    def forward(self, inputs):
        return self.segmentation_head(self.decoder(self.encoder(inputs)))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_data(tmp_path, positives=5):
    root, prepared = tmp_path / "raw", tmp_path / "prepared"
    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rgb[0, 0] = (0, 127, 255)
    for index in range(positives + 1):
        val = rgb.copy()
        if index < positives:
            val[1, :index % 4 + 1] = (0, 255, 255)
        for directory, pixels in (("org", rgb), ("val", val)):
            path = root / directory / "nested" / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path)
    prepare_gsi_dataset(root, prepared, gsi_category="paddy", oem_class_id=7,
                        label_color=(0, 255, 255))
    return root / "org", prepared


def test_dataset_rgb_labels_and_inference_preprocessing(tmp_path):
    org, prepared = make_data(tmp_path)
    samples, excluded = scan_dataset(org, prepared / "labels")
    assert len(samples) == 5 and len(excluded) == 1
    assert excluded[0].source_image_id == "nested/5"
    image, label = GsiPaddyDataset(samples)[0]
    assert image.shape == (3, 3, 5) and image.dtype == torch.float32
    assert image.min() == 0 and image.max() == 1
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([0, 127 / 255, 1]))
    assert label.dtype == torch.long and set(label.flatten().tolist()) == {7, 255}
    with Image.open(samples[0].image_path) as source:
        torch.testing.assert_close(inference.image_to_tensor(np.asarray(source))[0], image)


def test_572_padding_preserves_pixels_and_ignores_added_area():
    rgb = torch.rand(3, 572, 572)
    label = torch.full((572, 572), 255, dtype=torch.long)
    label[-1, -1] = 7
    images, labels = collate_padded([(rgb, label)])
    assert images.shape == (1, 3, 576, 576)
    torch.testing.assert_close(images[0, :, :572, :572], rgb)
    assert labels[0, 571, 571] == 7
    assert torch.all(labels[:, 572:] == 255) and torch.all(labels[:, :, 572:] == 255)
    torch.testing.assert_close(images[0, :, -1, -1], rgb[:, -1, -1])


@pytest.mark.parametrize("damage", ["missing", "invalid_label", "size"])
def test_reject_invalid_pairs(tmp_path, damage):
    org, prepared = make_data(tmp_path)
    path = prepared / "labels/nested/0.png"
    if damage == "missing":
        path.unlink()
    else:
        label = np.full((3, 5) if damage == "invalid_label" else (2, 5),
                        0 if damage == "invalid_label" else 7, dtype=np.uint8)
        Image.fromarray(label).save(path)
    with pytest.raises(ValueError):
        scan_dataset(org, prepared / "labels")


def test_deterministic_split_independent_of_input_order(tmp_path):
    org, prepared = make_data(tmp_path, positives=10)
    samples, _ = scan_dataset(org, prepared / "labels")
    train, val = split_samples(samples)
    assert (train, val) == split_samples(list(reversed(samples)), seed=42)
    assert len(train) == 8 and len(val) == 2
    assert set(train).isdisjoint(val) and set(train + val) == set(samples)
    assert (train, val) != split_samples(samples, seed=43)
    for ratio in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            split_samples(samples, ratio)
    with pytest.raises(ValueError, match="At least two"):
        split_samples(samples[:1])


def test_freeze_optimizer_and_bn_buffers_survive_training(tmp_path):
    org, prepared = make_data(tmp_path)
    samples, _ = scan_dataset(org, prepared / "labels")
    model = TinyModel()
    counts = training.freeze_encoder(model)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    assert all(p.requires_grad for p in model.segmentation_head.parameters())
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters())
    optimizer = training.make_optimizer(model, 1e-4)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(GsiPaddyDataset(samples), batch_size=2, collate_fn=collate_padded)
    metrics = training.run_epoch(model, loader, "cpu", optimizer)
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before["encoder." + key], rtol=0, atol=0)
    assert not model.encoder.training
    assert model.decoder.training and model.segmentation_head.training
    assert not torch.equal(before["segmentation_head.weight"], model.segmentation_head.weight)
    assert metrics["labeled_pixel_count"] == sum(s.positive_pixel_count for s in samples)


def test_ignore_pixels_have_no_loss_or_gradient_contribution():
    logits = torch.randn(1, 9, 2, 2, requires_grad=True)
    labels = torch.tensor([[[7, 255], [255, 255]]])
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(logits, labels)
    expected = nn.functional.cross_entropy(logits[:, :, 0, 0], torch.tensor([7]))
    torch.testing.assert_close(loss, expected)
    changed = logits.detach().clone()
    changed[:, :, 1, :] = 100
    changed[:, :, 0, 1] = -100
    torch.testing.assert_close(criterion(changed, labels), loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 1, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 0, 1]) == 0


def test_validation_metrics_are_pixel_weighted_and_ignore_unknowns():
    class PixelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Identity()

        def forward(self, images):
            logits = images.new_zeros((images.shape[0], 9, *images.shape[2:]))
            logits[:, 7] = images[:, 0]
            return logits

    model = PixelModel()
    first = torch.full((1, 3, 2, 2), 100.0)  # Ignored pixels deliberately differ.
    first[:, :, 0, 0] = 2
    batch = [(first, torch.tensor([[[7, 255], [255, 255]]])),
             (torch.zeros(1, 3, 2, 2), torch.full((1, 2, 2), 7))]
    metrics = training.run_epoch(model, batch, "cpu")
    assert metrics["labeled_pixel_count"] == 5
    assert metrics["class_7_labeled_pixel_agreement_recall"] == 1 / 5
    probability = np.exp(2) / (np.exp(2) + 8)
    assert metrics["loss"] == pytest.approx((-np.log(probability) + 4 * np.log(9)) / 5)
    assert metrics["class_7_mean_probability_on_labeled_pixels"] == pytest.approx((probability + 4 / 9) / 5)
    with pytest.raises(ValueError, match="All-ignore"):
        training.run_epoch(model, [(batch[0][0], torch.full((1, 2, 2), 255))], "cpu")


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_loader_matches_inference_and_architecture(tmp_path, monkeypatch, wrapped):
    import segmentation_models_pytorch as smp
    state = TinyModel().state_dict()
    path = tmp_path / "base.pth"
    torch.save({"state_dict": state} if wrapped else state, path)
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return TinyModel()
    monkeypatch.setattr(smp, "Unet", factory)
    loaded = inference.build_model(path)
    assert not loaded.training
    assert calls == [{"encoder_name": "efficientnet-b4", "encoder_weights": None,
                      "in_channels": 3, "classes": 9, "activation": None,
                      "decoder_attention_type": "scse"}]
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[key])


def test_synthetic_end_to_end_and_checkpoint_inference_reload(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kwargs: TinyModel())
    org, prepared = make_data(tmp_path)
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared),
        "--base-model", str(base), "--output-dir", str(tmp_path / "runs"),
        "--epochs", "2", "--num-threads", "1",
    ])
    run = training.train(args)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["total_images"] == 6 and manifest["usable_positive_images"] == 5
    assert manifest["excluded_all_ignore_images"] == 1
    assert manifest["train_count"] == 4 and manifest["validation_count"] == 1
    assert manifest["base_model_sha256"] == training._sha256(base)
    assert manifest["prepared_manifest_sha256"] == training._sha256(prepared / "manifest.json")
    assert len(manifest["epoch_metrics"]) == 2
    assert manifest["best_validation_loss"] == min(m["validation"]["loss"] for m in manifest["epoch_metrics"])
    assert (run / ".gitignore").read_text() == "*\n"
    for field in ("train_image_ids", "validation_image_ids", "excluded_image_ids", "dataset_inventory"):
        assert (run / manifest[field]).is_file()
    checkpoints = [m["checkpoint"] for m in manifest["epoch_metrics"]]
    checkpoints += [manifest["best_checkpoint"], manifest["final_checkpoint"]]
    for checkpoint in checkpoints:
        loaded = inference.build_model(run / checkpoint)
        assert loaded(torch.zeros(1, 3, 32, 32)).shape == (1, 9, 32, 32)
    assert training._sha256(run / manifest["final_checkpoint"]) == training._sha256(run / checkpoints[1])
    repeat = training.train(args)
    again = json.loads((repeat / "run_manifest.json").read_text())
    assert manifest["epoch_metrics"] == again["epoch_metrics"]
    assert (run / "train_ids.json").read_bytes() == (repeat / "train_ids.json").read_bytes()


def test_wrong_prepared_manifest_is_rejected_before_output(tmp_path):
    org, prepared = make_data(tmp_path)
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["gsi_category"] = "tree"
    training.write_json(path, manifest)
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    output = tmp_path / "runs"
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared), "--base-model", str(base),
        "--output-dir", str(output),
    ])
    with pytest.raises(ValueError, match="must specify paddy"):
        training.train(args)
    assert not output.exists()


def test_failed_run_is_marked_without_final_checkpoint(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp, "Unet", lambda **kwargs: TinyModel())
    org, prepared = make_data(tmp_path)
    base = tmp_path / "base.pth"
    torch.save(TinyModel().state_dict(), base)
    output = tmp_path / "runs"
    args = training.make_parser().parse_args([
        "--org-dir", str(org), "--prepared-dir", str(prepared),
        "--base-model", str(base), "--output-dir", str(output),
    ])
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(training, "run_epoch", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        training.train(args)
    run = next(output.iterdir())
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["failure_type"] == "RuntimeError"
    assert manifest["final_checkpoint"] is None
    assert not (run / "checkpoints/final.pth").exists()
