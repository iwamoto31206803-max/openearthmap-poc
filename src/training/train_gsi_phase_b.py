"""Phase B v0.1: add Water-positive supervision to the Phase A v0.3 objective."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
PADDY_COUNTS = (1286, 1314)
WATER_COUNTS = (692, 558)


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
                      alpha=1.0):
    """Run one Paddy-axis epoch; optional Water never creates another optimizer update."""
    replay.validate_replay(lambda_preserve, temperature, alpha)
    training = optimizer is not None
    model.train(training); model.encoder.eval(); teacher.requires_grad_(False); teacher.eval()
    schedule = schedule or set()
    sources = {name: {"ce": 0., "kl": 0., "p": 0, "u": 0, "agree": 0,
                             "prob": 0., "base_agree": 0}
               for name in ("paddy", "water")}
    replay_sum = replay_pixels = replay_batches = steps = water_batches = 0
    replay_iter, water_iter = iter(replay_loader), iter(water_loader)
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
                parts.append(w[0]); batches.append(("water", w, WATER_CLASS)); water_batches += 1
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
    pm, wm = metrics("paddy", 7), metrics("water", 6)
    replay_kl = replay_sum/replay_pixels
    loss = pm["positive_ce"] + lambda_preserve*pm["unknown_preservation_kl"] + alpha*lambda_preserve*replay_kl
    if wm: loss += wm["positive_ce"] + lambda_preserve*wm["unknown_preservation_kl"]
    return {"loss": loss, "paddy": pm, "water": wm,
            "replay": {"preservation_kl": replay_kl, "pixel_count": replay_pixels,
                       "batch_count": replay_batches},
            "optimizer_update_count": steps if training else 0, "logical_step_count": steps,
            "water_step_count": water_batches}


def validate_phase_b(model, paddy_loader, water_loader, replay_loader, device, *, teacher,
                     lambda_preserve=1.0, temperature=1.0, alpha=1.0):
    """Evaluate every validation sample once and report each source separately."""
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
    rkl=replay_sum/pixels
    loss=sum((paddy["positive_ce"],lambda_preserve*paddy["unknown_preservation_kl"],
              water["positive_ce"],lambda_preserve*water["unknown_preservation_kl"],alpha*lambda_preserve*rkl))
    return {"loss":loss,"paddy":paddy,"water":water,
            "replay":{"preservation_kl":rkl,"student_base_argmax_agreement":agreed/pixels,"pixel_count":pixels}}


def _audit_manifest(path, category, target):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gsi_category") != category or value.get("oem_class_id") != target or value.get("ignore_index") != 255:
        raise ValueError(f"Prepared manifest must specify {category}, class {target}, ignore 255")
    return value


def train(args):
    if args.seed != 42 or args.train_ratio != .8 or args.batch_size != 1:
        raise ValueError("Phase B v0.1 requires seed=42, train-ratio=0.8, batch-size=1")
    epochs = 1 if args.epochs is None else args.epochs
    if epochs < 1 or args.num_threads < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid training settings")
    replay.validate_replay(args.lambda_preserve, args.temperature, args.alpha_replay)
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
    model = build_model(base, device=device); counts = freeze_encoder(model); optimizer = make_optimizer(model, args.learning_rate)
    teacher = build_teacher(base, device); replay.check_model_invariants(model, teacher, optimizer, initial=True)
    if _sha256(base) != base_sha: raise ValueError("Base checkpoint changed during loading")
    timestamp = datetime.now(timezone.utc); run_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")+"_"+uuid.uuid4().hex[:8]
    run_dir = (args.output_dir or Path("training_outputs/gsi_phase_b_v01")).resolve()/run_id
    run_dir.mkdir(parents=True); (run_dir/".gitignore").write_text("*\n"); (run_dir/"checkpoints").mkdir()
    files = {"paddy_train_ids": pt, "paddy_validation_ids": pv, "paddy_replay_train_ids": rt,
             "paddy_replay_validation_ids": rv, "water_positive_train_ids": wt,
             "water_positive_validation_ids": wv}
    for name, samples in files.items(): write_json(run_dir/(name+".json"), [s.source_image_id for s in samples])
    schedule = water_schedule(len(pt), len(wt), 42)
    manifest = {"schema_version": 4, "experiment": "gsi_phase_b_v0.1", "training_mode": "phase_b_v0.1",
        "status": "running", "run_id": run_id, "timestamp_utc": timestamp.isoformat(),
        "base_model_path": str(base), "base_model_sha256": base_sha,
        "student_initialization_checkpoint_sha256": base_sha, "teacher_checkpoint_sha256": base_sha,
        "original_base_start": True, "git_commit_sha": replay.current_git_commit(), "seed": 42,
        "lambda_preserve": args.lambda_preserve, "temperature": args.temperature, "alpha_replay": args.alpha_replay,
        "epochs": epochs, "batch_size": 1, "learning_rate": args.learning_rate,
        "optimizer": {"name":"AdamW","weight_decay":.01,"betas":[.9,.999],"eps":1e-8},
        "architecture":{"library":"segmentation_models_pytorch","model":"Unet",**MODEL_KWARGS}, "preprocessing":PREPROCESSING,
        "parameter_counts": counts, "paddy_target_class":7, "water_target_class":6,
        "paddy_train_count":len(pt), "paddy_validation_count":len(pv), "paddy_replay_train_count":len(rt),
        "paddy_replay_validation_count":len(rv), "water_positive_train_count":len(wt),
        "water_positive_validation_count":len(wv), "water_all_ignore_count":len(water_ignored),
        "water_all_ignore_usage":"not used for training, replay, or validation",
        "total_optimizer_updates":epochs*len(pt), "epoch_axis":"Paddy positive train split",
        "split_method":"sort IDs, random.Random(seed).shuffle, floor(n*ratio), clamp to [1,n-1]",
        "water_schedule":{"deterministic":True,"seed":42,"method":"random.sample slot set; consume next Water train sample at each selected slot",
                          "slot_count":len(pt),"water_slot_count":len(wt)},
        "loss":"L_paddy + optional L_water + alpha_replay * L_replay; one backward and optimizer step",
        "prepared_manifest_sha256":{"paddy":_sha256(pp),"water":_sha256(wp)},
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
    try:
        if args.preflight:
            manifest["preflight_metrics"] = run_phase_b_epoch(model, islice(pl,1), device, teacher=teacher,
                replay_loader=list(islice(rl,1)), water_loader=list(islice(wl,1)), schedule={0},
                lambda_preserve=args.lambda_preserve, temperature=args.temperature, alpha=args.alpha_replay)
            manifest["status"]="preflight_passed"; return run_dir
        for epoch in range(1, epochs+1):
            train_schedule = {0} if args.smoke_test else schedule
            training = run_phase_b_epoch(model, islice(pl,1) if args.smoke_test else pl, device, optimizer,
                teacher=teacher,replay_loader=list(islice(rl,1)) if args.smoke_test else rl,
                water_loader=list(islice(wl,1)) if args.smoke_test else wl,schedule=train_schedule,
                lambda_preserve=args.lambda_preserve,temperature=args.temperature,alpha=args.alpha_replay)
            validation = validate_phase_b(model,pvl,wvl,rvl,device,teacher=teacher,
                lambda_preserve=args.lambda_preserve,temperature=args.temperature,alpha=args.alpha_replay)
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
    parser.add_argument("--base-model",type=Path,default=default_model_path()); parser.add_argument("--base-sha256")
    parser.add_argument("--output-dir",type=Path); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--train-ratio",type=float,default=.8); parser.add_argument("--epochs",type=int,default=None)
    parser.add_argument("--batch-size",type=int,default=1); parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--lambda-preserve",type=float,default=1.); parser.add_argument("--temperature",type=float,default=1.)
    parser.add_argument("--alpha-replay",type=float,default=1.); parser.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    parser.add_argument("--num-threads",type=int,default=2)
    checks=parser.add_mutually_exclusive_group(); checks.add_argument("--preflight",action="store_true"); checks.add_argument("--smoke-test",action="store_true")
    return parser


if __name__ == "__main__": train(make_parser().parse_args())
