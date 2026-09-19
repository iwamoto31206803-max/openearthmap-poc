"""Synthetic-only tests for Phase B v0.1; no real GSI data or Base weights."""

import copy
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest
import torch

from src.training import train_gsi_phase_b as phase_b
from src.training.prepare_gsi_labels import _sha256
from test_replay_preservation import batches, make_replay_data
from test_train_gsi_paddy import TinyModel, make_data, single_thread  # noqa: F401


def water_batch():
    images=torch.rand(1,3,3,5); labels=torch.full((1,3,5),255); labels[:,0,0]=6
    return images, labels, torch.ones_like(labels,dtype=torch.bool)


def road_batch():
    images=torch.rand(1,3,3,5); labels=torch.full((1,3,5),255); labels[:,0,0]=4
    return images, labels, torch.ones_like(labels,dtype=torch.bool)


def test_water_mask_ce_and_unknown_preservation_are_disjoint():
    student=torch.randn(1,9,2,2,requires_grad=True); teacher=torch.randn(1,9,2,2,requires_grad=True)
    labels=torch.tensor([[[6,255],[255,255]]]); mask=torch.ones_like(labels,dtype=torch.bool)
    total, ce, kl=phase_b.positive_preservation_losses(student,teacher,labels,mask,6)
    torch.testing.assert_close(ce,torch.nn.functional.cross_entropy(student[:,:,0,0],torch.tensor([6])))
    ce_grad=torch.autograd.grad(ce,student,retain_graph=True)[0]
    kl_grad=torch.autograd.grad(kl,student,retain_graph=True)[0]
    assert torch.count_nonzero(ce_grad[:,:,0,1:]) == 0 and torch.count_nonzero(ce_grad[:,:,1]) == 0
    assert torch.count_nonzero(kl_grad[:,:,0,0]) == 0
    assert torch.count_nonzero(kl_grad[:,:,0,1:])+torch.count_nonzero(kl_grad[:,:,1]) > 0
    total.backward(); assert teacher.grad is None


def test_deterministic_dispersed_water_schedule():
    first=phase_b.water_schedule(10,5,42)
    assert isinstance(first, set)
    assert first == phase_b.water_schedule(10,5,42)
    assert first != phase_b.water_schedule(10,5,43)
    assert len(first)==5
    assert first != set(range(5))
    with pytest.raises(ValueError): phase_b.water_schedule(2,3)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_beta_water_rejected(value):
    with pytest.raises(ValueError, match="beta_water"):
        phase_b.validate_beta_water(value)


def test_beta_water_cli_default_preserves_v01():
    parser = phase_b.make_parser()
    required = ["--org-dir", "p", "--prepared-dir", "p", "--water-org-dir", "w",
                "--water-prepared-dir", "w"]
    assert parser.parse_args(required).beta_water == 1.0
    assert parser.parse_args(required + ["--beta-water", "0.5"]).beta_water == 0.5
    assert parser.parse_args(required).beta_road == 1.0


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_beta_road_rejected(value):
    with pytest.raises(ValueError, match="beta_road"):
        phase_b.validate_beta_road(value)


@pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
def test_beta_road_scales_only_road_objective_and_metrics_stay_raw(beta):
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    paddy,replay_batch=batches(); road=road_batch()
    common=dict(teacher=teacher,replay_loader=[replay_batch],water_loader=[],schedule=set(),
                road_loader=[road])
    full=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_road=1.0,**common)
    weighted=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_road=beta,**common)
    objective=full["road"]["positive_ce"]+full["road"]["unknown_preservation_kl"]
    assert weighted["loss"] == pytest.approx(full["loss"]-(1-beta)*objective)
    assert weighted["paddy"] == full["paddy"]
    assert weighted["water"] == full["water"] is None
    assert weighted["replay"] == full["replay"] and weighted["road"] == full["road"]
    if beta == 0:
        images,labels,mask=road
        road_loss,_,_=phase_b.positive_preservation_losses(
            student(images),teacher(images),labels,mask,phase_b.ROAD_CLASS)
        gradients=torch.autograd.grad(beta*road_loss,tuple(student.parameters()),allow_unused=True)
        assert all(value is None or torch.count_nonzero(value)==0 for value in gradients)


def test_no_water_step_keeps_road_paddy_and_replay_in_one_update():
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    from src.training.train_gsi_paddy import freeze_encoder, make_optimizer
    freeze_encoder(student); optimizer=make_optimizer(student,1e-4); updates=[]
    optimizer.register_step_post_hook(lambda *args: updates.append(1))
    paddy,replay_batch=batches()
    result=phase_b.run_phase_b_epoch(student,[paddy],"cpu",optimizer,teacher=teacher,
        replay_loader=[replay_batch],water_loader=[],schedule=set(),road_loader=[road_batch()])
    assert len(updates)==result["logical_step_count"]==result["optimizer_update_count"]==1
    assert result["water_step_count"]==0 and result["road_step_count"]==1
    assert result["water"] is None and result["road"] is not None


def test_beta_scales_only_complete_water_objective():
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    paddy,replay_batch=batches(); water=water_batch()
    common=dict(teacher=teacher,replay_loader=[replay_batch],water_loader=[water],schedule={0})
    full=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_water=1.0,**common)
    half=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_water=0.5,**common)
    water_objective=full["water"]["positive_ce"]+full["water"]["unknown_preservation_kl"]
    assert half["loss"] == pytest.approx(full["loss"] - 0.5*water_objective)
    assert half["paddy"] == full["paddy"] and half["replay"] == full["replay"]
    assert half["water"] == full["water"]  # source metrics remain raw


def test_beta_has_no_effect_without_water_and_zero_removes_water_gradient():
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    paddy,replay_batch=batches()
    common=dict(teacher=teacher,replay_loader=[replay_batch],water_loader=[],schedule=set())
    zero=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_water=0.0,**common)
    large=phase_b.run_phase_b_epoch(student,[paddy],"cpu",beta_water=9.0,**common)
    assert zero["loss"] == pytest.approx(large["loss"])

    images,labels,mask=water_batch(); logits=student(images); base=teacher(images)
    water_loss,_,_=phase_b.positive_preservation_losses(logits,base,labels,mask,6)
    gradients=torch.autograd.grad(0.0*water_loss,tuple(student.parameters()),allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)


def test_phase_b_inventory_regression():
    schedule=phase_b.water_schedule(1028,553,42)
    assert len(schedule) == 553 and min(schedule) >= 0 and max(schedule) < 1028


def test_road_pilot_sampling_is_deterministic_unique_and_exact():
    pool=list(range(1305))
    selected=phase_b.road_pilot_samples(pool,1028,42)
    assert selected == phase_b.road_pilot_samples(pool,1028,42)
    assert len(selected)==len(set(selected))==1028 and len(pool)-len(selected)==277
    with pytest.raises(ValueError): phase_b.road_pilot_samples(pool,1306,42)


def test_road_1028_steps_create_exactly_1028_optimizer_updates():
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    from src.training.train_gsi_paddy import freeze_encoder, make_optimizer
    freeze_encoder(student); optimizer=make_optimizer(student,1e-4); updates=[]
    optimizer.register_step_post_hook(lambda *args: updates.append(1))
    paddy,replay_batch=batches(); road=road_batch()
    result=phase_b.run_phase_b_epoch(student,[paddy]*1028,"cpu",optimizer,teacher=teacher,
        replay_loader=[replay_batch],water_loader=[],schedule=set(),road_loader=[road]*1028)
    assert len(updates)==result["logical_step_count"]==result["optimizer_update_count"]==1028
    assert result["road_step_count"]==1028


@pytest.mark.parametrize("with_water", [False, True])
def test_logical_step_composition_and_single_update(with_water):
    student=TinyModel(); teacher=copy.deepcopy(student).requires_grad_(False).eval()
    from src.training.train_gsi_paddy import freeze_encoder, make_optimizer
    freeze_encoder(student); optimizer=make_optimizer(student,1e-4); steps=[]
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    paddy,replay_batch=batches(); water=[water_batch()] if with_water else []
    result=phase_b.run_phase_b_epoch(student,[paddy]*3,"cpu",optimizer,teacher=teacher,
        replay_loader=[replay_batch],water_loader=water,schedule={1} if with_water else set())
    assert len(steps)==result["optimizer_update_count"]==result["logical_step_count"]==3
    assert result["water_step_count"] == int(with_water)
    assert (result["water"] is not None) is with_water
    expected=result["paddy"]["positive_ce"]+result["paddy"]["unknown_preservation_kl"]+result["replay"]["preservation_kl"]
    if with_water: expected += result["water"]["positive_ce"]+result["water"]["unknown_preservation_kl"]
    assert result["loss"] == pytest.approx(expected)


def make_water_data(tmp_path):
    org=tmp_path/"water_org"; prepared=tmp_path/"water_prepared"; (org/"nested").mkdir(parents=True); (prepared/"labels"/"nested").mkdir(parents=True)
    positives=3
    for i in range(5):
        Image.new("RGB",(5,3),(i,2,3)).save(org/"nested"/f"{i}.png")
        label=np.full((3,5),255,dtype=np.uint8)
        if i < positives: label[0,0]=6
        Image.fromarray(label).save(prepared/"labels"/"nested"/f"{i}.png")
    manifest={"gsi_category":"water","oem_class_id":6,"ignore_index":255,"image_count":5,
              "false_image_count":2,"positive_pixel_count":3,"total_pixel_count":75}
    (prepared/"manifest.json").write_text(json.dumps(manifest))
    return org,prepared


def make_road_data(tmp_path):
    org=tmp_path/"road_org"; labels=tmp_path/"road_prepared"/"labels"
    (org/"a").mkdir(parents=True); (labels/"a").mkdir(parents=True)
    for name, positive in (("one",True),("ignore",False)):
        Image.new("RGB",(3,2),(9,8,7)).save(org/"a"/f"{name}.png")
        value=np.full((2,3),255,dtype=np.uint8)
        if positive: value[0,0]=4
        Image.fromarray(value).save(labels/"a"/f"{name}.png")
    return org, labels


def test_road_scanner_separates_all_ignore_and_rejects_invalid(tmp_path):
    org,labels=make_road_data(tmp_path)
    positive,ignored=phase_b.scan_road_dataset(org,labels)
    assert len(positive)==len(ignored)==1 and positive[0].positive_pixel_count==1
    invalid=np.full((2,3),255,dtype=np.uint8); invalid[0,0]=6
    Image.fromarray(invalid).save(labels/"a"/"one.png")
    with pytest.raises(ValueError,match="only 4 and 255"):
        phase_b.scan_road_dataset(org,labels)


def test_road_overlap_uses_content_hash_not_filename(tmp_path):
    org,labels=make_road_data(tmp_path); road,_=phase_b.scan_road_dataset(org,labels)
    shared=replace(road[0],source_image_id="totally/different/name")
    other_hash="0"*64
    other_shared=replace(road[0],source_image_id="another/name",image_sha256=other_hash)
    unique=replace(road[0],source_image_id="same-looking-name",image_sha256="1"*64)
    paddy=[road[0],replace(road[0],source_image_id="paddy/other",image_sha256=other_hash)]
    retained,excluded,references=phase_b.exclude_source_hash_overlap(
        [shared,other_shared,unique],paddy)
    assert retained==[unique] and excluded==[other_shared,shared]
    assert references == [phase_b._road_overlap_reference(other_hash),
                          phase_b._road_overlap_reference(shared.image_sha256)]
    assert references[0] != references[1]
    assert all(reference.startswith("overlap-") and len(reference)==24 for reference in references)
    assert all(raw_hash not in reference
               for raw_hash in (other_hash,shared.image_sha256) for reference in references)


def test_road_overlap_anonymous_reference_is_stable_across_names_and_runs():
    raw_hash="a"*64
    first=phase_b._road_overlap_reference(raw_hash)
    assert first == phase_b._road_overlap_reference(raw_hash)
    assert first != phase_b._road_overlap_reference("b"*64)
    assert raw_hash not in first


@pytest.mark.parametrize(("beta", "experiment"), [(1.0, "gsi_phase_b_v0.1"),
                                                   (0.5, "gsi_phase_b_v0.2")])
def test_water_all_ignore_excluded_and_phase_b_smoke_starts_from_base(
        tmp_path, monkeypatch, beta, experiment):
    import segmentation_models_pytorch as smp
    monkeypatch.setattr(smp,"Unet",lambda **kwargs:TinyModel())
    monkeypatch.setattr(phase_b,"PADDY_COUNTS",(5,4)); monkeypatch.setattr(phase_b,"WATER_COUNTS",(3,2))
    org,prepared=make_replay_data(tmp_path); water_org,water_prepared=make_water_data(tmp_path)
    base=tmp_path/"base.pth"; torch.save(TinyModel().state_dict(),base)
    args=phase_b.make_parser().parse_args(["--org-dir",str(org),"--prepared-dir",str(prepared),
        "--water-org-dir",str(water_org),"--water-prepared-dir",str(water_prepared),
        "--base-model",str(base),"--base-sha256",_sha256(base),"--output-dir",str(tmp_path/"runs"),
        "--num-threads","1","--beta-water",str(beta),"--smoke-test"])
    run=phase_b.train(args); manifest=json.loads((run/"run_manifest.json").read_text())
    assert manifest["experiment"]==experiment and manifest["original_base_start"] is True
    assert manifest["model_version"]==experiment and manifest["beta_water"]==beta
    assert manifest["student_initialization_checkpoint_sha256"]==manifest["teacher_checkpoint_sha256"]==_sha256(base)
    assert manifest["water_positive_train_count"]==2 and manifest["water_positive_validation_count"]==1
    assert manifest["water_all_ignore_count"]==2 and manifest["water_all_ignore_usage"].startswith("not used")
    water_ids=json.loads((run/manifest["id_files"]["water_positive_train_ids"]["path"]).read_text())
    assert len(water_ids)==2 and all(int(value.split("/")[-1]) < 3 for value in water_ids)
    metrics=manifest["epoch_metrics"][0]
    assert metrics["training"]["optimizer_update_count"]==1
    assert set(metrics["validation"]) >= {"paddy","water","replay"}
