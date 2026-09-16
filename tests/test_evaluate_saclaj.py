"""Offline tests: synthetic sites and models, user-confirmed SACLAJ category IDs."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image
import pytest
import torch

from src import model as shared_model, predict_geotiff_tiled
from src.evaluation import saclaj as data
from src.evaluation import evaluate_saclaj as evaluation

TEMPLATE = Path(__file__).resolve().parents[1] / "docs/saclaj_mapping.template.json"
CATEGORY_SUBTYPES = {
    "9": "rice_paddy", "11": "other_crop", "18": "mixed", "19": "needleleaf",
    "20": "broadleaf", "22": "needleleaf_evergreen", "23": "broadleaf_evergreen",
    "27": "needleleaf_deciduous", "28": "broadleaf_deciduous", "33": "water",
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests must never use network")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def mapping_document():
    table = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    table.update(definition_source="SYNTHETIC PROVENANCE ONLY", definition_sha256="a" * 64)
    return table


def sites_per_category(n=1):
    return [data.Site(f"synthetic-private-{subtype}-{i}", category_id,
                      35.123456, 139.234567, "2020/01/01", 9999.0, subtype)
            for category_id, subtype in CATEGORY_SUBTYPES.items() for i in range(n)]


def write_csv(path, sites=None, diameter_column="Diameter（m）", extra_empty_column=False):
    rows = []
    for site in sites if sites is not None else sites_per_category():
        rows.append({"ID": site.site_id, "Category_ID": site.category_id,
                     "Latitude": site.latitude, "Longitude": site.longitude,
                     "Date": site.date, diameter_column: site.diameter,
                     "Category_detail": site.detail, "Note": "PRIVATE NOTE MUST NOT BE EXPORTED"})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [diameter_column if name == "Diameter（m）" else name for name in data.REQUIRED_COLUMNS]
        if extra_empty_column:
            fields.append("")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("diameter_column", ["Diameter（m）", "Diameter(m)"])
def test_schema_and_bom(tmp_path, diameter_column):
    path = write_csv(tmp_path / "input.csv", diameter_column=diameter_column, extra_empty_column=True)
    loaded = data.read_csv(path)
    assert loaded.sites == sites_per_category()
    assert loaded.diameter_column == diameter_column
    site = replace(sites_per_category()[0], diameter=None)
    assert data.read_csv(write_csv(path, [site], diameter_column=diameter_column)).sites == [site]


@pytest.mark.parametrize("columns", [[], ["Diameter（m）", "Diameter(m)"]])
def test_missing_or_ambiguous_diameter_headers_fail(tmp_path, columns):
    path = tmp_path / "input.csv"
    fields = [name for name in data.REQUIRED_COLUMNS if name != "Diameter（m）"] + columns
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(fields)
    with pytest.raises(data.ValidationError, match="exactly one Diameter"):
        data.read_csv(path)


@pytest.mark.parametrize("damage", ["header", "duplicate_header", "duplicate_id", "empty_id",
                                       "category", "latitude", "longitude", "nan", "diameter", "date", "extra", "missing"])
def test_schema_fail_fast_without_values(tmp_path, damage):
    path = write_csv(tmp_path / "input.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if damage == "header":
        rows[0][0] = "id"
    elif damage == "duplicate_header":
        rows[0][-1] = "ID"
    elif damage == "duplicate_id":
        rows[2][0] = rows[1][0]
    elif damage == "extra":
        rows[1].append("private")
    elif damage == "missing":
        rows[1].pop()
    else:
        field, value = {"empty_id": (0, ""), "category": (1, "1.5"),
                        "latitude": (2, "91"), "longitude": (3, "181"),
                        "nan": (2, "nan"), "diameter": (5, "-1"), "date": (4, "")}[damage]
        rows[1][field] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError) as caught:
        data.read_csv(path)
    assert "synthetic-private" not in str(caught.value)


def test_category_id_mapping_ignores_detail_and_excludes_unregistered(tmp_path):
    path = tmp_path / "mapping.json"
    document = mapping_document()
    path.write_text(json.dumps(document))
    table = data.load_mapping(path)
    selected, counts = data.select_sites(sites_per_category() + [
        replace(sites_per_category()[0], site_id="free-form", detail="rice_paddy or urban"),
        replace(sites_per_category()[0], site_id="empty-detail", detail=""),
        replace(sites_per_category()[0], site_id="unknown", category_id="0"),
        *[replace(sites_per_category()[0], site_id=f"excluded-{category}", category_id=category,
                  detail="rice_paddy") for category in ("8", "10", "17", "32")],
    ], table)
    assert len(selected) == 12
    assert counts["excluded_unmapped_or_ambiguous"] == 5
    assert {r.site.site_id for r in selected} >= {"free-form", "empty-detail"}
    assert set(r.subtype for r in selected) == set(data.SUBTYPES)
    for ref in selected:
        assert ref.subtype == CATEGORY_SUBTYPES[ref.site.category_id]
        assert ref.expected_class == (7 if ref.site.category_id in {"9", "11"}
                                      else 6 if ref.site.category_id == "33" else 5)
    for bad in (False, None):
        document["definition_verified"] = bad
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="verified"):
            data.load_mapping(path)
    document = mapping_document()
    document["entries"].append(document["entries"][0])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Duplicate"):
        data.load_mapping(path)


def test_mapping_template_matches_review_and_requires_readme_hash(tmp_path):
    table = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert table["schema_version"] == 2 and table["definition_verified"] is True
    assert {e["category_id"]: e["subtype"] for e in table["entries"]} == {
        **CATEGORY_SUBTYPES, "8": None, "10": None, "17": None, "32": None,
    }
    assert all(set(e) == {"category_id", "subtype"} for e in table["entries"])
    with pytest.raises(data.ValidationError, match="SHA256"):
        data.load_mapping(TEMPLATE)  # Do not fabricate the user's local README hash.
    path = tmp_path / "mapping.json"
    for damage in ("old_schema", "detail", "unverified", "hash"):
        table = mapping_document()
        if damage == "old_schema":
            table["schema_version"] = 1
        elif damage == "detail":
            table["entries"][0]["category_detail"] = "not a mapping key"
        elif damage == "unverified":
            table["definition_verified"] = False
        else:
            table["definition_sha256"] = "invalid"
        path.write_text(json.dumps(table))
        with pytest.raises(data.ValidationError):
            data.load_mapping(path)


@pytest.mark.parametrize("lat,lon,expected", [(20, 122, True), (46, 154, True), (35, 139, True),
    (19.99, 139, False), (46.01, 139, False), (35, 121.99, False), (35, 154.01, False)])
def test_japan_bbox(lat, lon, expected):
    assert data.in_japan(replace(sites_per_category()[0], latitude=lat, longitude=lon)) is expected


def test_deterministic_stratified_sampling_and_maximum():
    sites = sites_per_category(120)
    selected, counts = data.select_sites(sites, mapping_document())
    reverse, _ = data.select_sites(sites[::-1], mapping_document())
    assert selected == reverse and len(selected) == 1000
    assert set(counts["sampled_by_subtype"].values()) == {100}
    assert set(counts["before_sampling_by_subtype"].values()) == {120}
    assert selected != data.select_sites(sites, mapping_document(), seed=43)[0]
    changed_detail, _ = data.select_sites([replace(site, detail="自由記述") for site in sites], mapping_document())
    assert [ref.site.site_id for ref in selected] == [ref.site.site_id for ref in changed_detail]
    assert len(data.select_sites(sites, mapping_document(), maximum=1)[0]) == 10
    with pytest.raises(ValueError):
        data.select_sites(sites, mapping_document(), maximum=0)


def result(subtype, base_class, ft_class, base_probability=0.2, ft_probability=0.8):
    return {"ID": "SECRET-ID", "Latitude": 35.123456, "Longitude": 139.234567,
            "subtype": subtype, "expected_class": data.SUBTYPES[subtype], "status": "success",
            "base": {"predicted_class": base_class, "expected_probability": base_probability,
                     "agriculture_probability": 0.1},
            "fine_tuned": {"predicted_class": ft_class, "expected_probability": ft_probability,
                           "agriculture_probability": 0.7}}


def test_paired_metrics_retention_probability_and_sanitization():
    rows = [result("rice_paddy", 7, 7), result("rice_paddy", 5, 7),
            result("broadleaf", 5, 7), result("needleleaf", 7, 5),
            result("mixed", 5, 5), result("water", 6, 7), result("other_crop", 7, 7)]
    rows.append({"subtype": "rice_paddy", "expected_class": 7, "status": "skipped", "reason": "SECRET"})
    counts = data.select_sites(sites_per_category(2), mapping_document())[1]
    counts["ID"] = "SECRET"
    summary = data.aggregate(rows, counts)
    assert summary["rice_agreement_percent"] == {"n": 2, "base": 50.0, "fine_tuned": 100.0, "delta_pp": 50.0}
    assert summary["agriculture_leakage_percent"] == {"n": 4, "base": 25.0, "fine_tuned": 50.0, "delta_pp": 25.0}
    for subtype, delta in (("broadleaf", -100), ("needleleaf", 100), ("mixed", 0), ("water", -100)):
        assert summary["categories"][subtype]["agreement_percent"]["delta_pp"] == delta
    rice = summary["categories"]["rice_paddy"]
    assert rice["mean_expected_probability"]["delta"] == pytest.approx(0.6)
    assert rice["mean_agriculture_probability"]["delta"] == pytest.approx(0.6)
    serialized = json.dumps(summary)
    for forbidden in ("SECRET", "Latitude", "Longitude", '"ID"', "35.123456", "139.234567"):
        assert forbidden not in serialized
    empty = data.aggregate([], counts)
    assert empty["rice_agreement_percent"]["base"] is None
    assert empty["agriculture_leakage_percent"]["delta_pp"] is None


def test_evergreen_and_deciduous_subtypes_have_separate_metrics():
    subtypes = ["needleleaf_evergreen", "broadleaf_evergreen", "needleleaf_deciduous", "broadleaf_deciduous"]
    rows = [result(subtype, base, ft) for subtype, base, ft in zip(subtypes, [5, 7, 5, 7], [7, 5, 5, 7])]
    summary = data.aggregate(rows, data.select_sites(sites_per_category(), mapping_document())[1])
    for subtype, delta in zip(subtypes, [-100, 100, 0, 0]):
        category = summary["categories"][subtype]
        assert category["expected_class"] == 5
        assert category["agreement_percent"]["n"] == 1
        assert category["agreement_percent"]["delta_pp"] == delta
    assert summary["categories"]["needleleaf"]["agreement_percent"]["n"] == 0
    assert summary["categories"]["broadleaf"]["agreement_percent"]["n"] == 0
    assert summary["agriculture_leakage_percent"]["n"] == 4


class TinyPointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.arange(9, dtype=torch.float32))
        self.inputs = []

    def forward(self, tensor):
        self.inputs.append(tensor.detach().clone())
        return self.values[None, :, None, None].expand(tensor.shape[0], 9, *tensor.shape[2:])


def test_preprocessing_same_tensor_center_pixel_and_diameter_unused():
    patch = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
    class SpatialModel(TinyPointModel):
        def forward(self, tensor):
            logits = super().forward(tensor).clone()
            logits[:, 7, 256, 256] = 20
            return logits
    base, ft = SpatialModel(), SpatialModel()
    pair = evaluation.predict_pair(patch, base, ft, 7, "cpu")
    assert pair["base"] == pair["fine_tuned"] and pair["base"]["predicted_class"] == 7
    torch.testing.assert_close(base.inputs[0], ft.inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(base.inputs[0], predict_geotiff_tiled.image_to_tensor(patch), rtol=0, atol=0)
    assert base.inputs[0].dtype == torch.float32


def test_patch_alignment_crosses_tiles_without_resampling(monkeypatch):
    monkeypatch.setattr(evaluation, "lonlat_to_tile", lambda *args: (10 + 255.75 / 256, 20 + 1.25 / 256))
    calls = []
    def tile(z, x, y):
        calls.append((z, x, y))
        array = np.zeros((256, 256, 3), dtype=np.uint8)
        array[:, :, 0] = x
        array[:, :, 1] = y
        array[:, :, 2] = np.arange(256, dtype=np.uint8)
        return Image.fromarray(array)
    monkeypatch.setattr(evaluation, "download_tile", tile)
    site = sites_per_category()[0]
    patch = evaluation.acquire_patch(site)
    assert patch.shape == (512, 512, 3) and len(calls) == 9
    assert patch[256, 256].tolist() == [10, 20, 255]
    assert patch[256, 257].tolist() == [11, 20, 0]
    np.testing.assert_array_equal(evaluation.acquire_patch(replace(site, diameter=0)), patch)


@pytest.mark.parametrize("failure,status,reason", [
    (HTTPError("SECRET-URL", 404, "missing", {}, None), "skipped", "imagery_unavailable"),
    (HTTPError("SECRET-URL", 503, "server", {}, None), "error", "http_error"),
    (URLError("SECRET-URL"), "error", "network_failure"),
])
def test_acquisition_failure_is_private(monkeypatch, failure, status, reason):
    def fail(*args):
        raise failure
    monkeypatch.setattr(evaluation, "download_tile", fail)
    with pytest.raises(evaluation.AcquisitionFailure) as caught:
        evaluation.acquire_patch(sites_per_category()[0])
    assert (caught.value.status, caught.value.reason) == (status, reason)
    assert "SECRET" not in str(caught.value)


def setup_run(tmp_path, monkeypatch):
    import segmentation_models_pytorch as smp
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return TinyPointModel()
    monkeypatch.setattr(smp, "Unet", factory)
    csv_path = write_csv(tmp_path / "private.csv")
    table = tmp_path / "mapping.json"
    table.write_text(json.dumps(mapping_document()))
    for name, predicted in (("base", 5), ("ft", 7)):
        model = TinyPointModel()
        with torch.no_grad():
            model.values[predicted] = 20
        torch.save(model.state_dict(), tmp_path / f"{name}.pth")
    args = evaluation.make_parser().parse_args([
        "--saclaj-csv", str(csv_path), "--mapping", str(table),
        "--base-model", str(tmp_path / "base.pth"), "--fine-tuned-model", str(tmp_path / "ft.pth"),
        "--output-dir", str(tmp_path / "results"), "--num-threads", "1",
    ])
    return args, calls


def test_synthetic_end_to_end_skips_and_manifest(tmp_path, monkeypatch):
    args, loaders = setup_run(tmp_path, monkeypatch)
    write_csv(args.saclaj_csv, [replace(site, detail="" if site.category_id == "9" else "補足情報")
                               for site in sites_per_category()], extra_empty_column=True)
    acquired = []
    def acquire(site, zoom):
        acquired.append(site.site_id)
        if site.category_id == "18":
            raise evaluation.AcquisitionFailure("skipped", "imagery_unavailable")
        if site.category_id == "33":
            raise evaluation.AcquisitionFailure("error", "network_failure")
        return np.full((512, 512, 3), 127, dtype=np.uint8)
    monkeypatch.setattr(evaluation, "acquire_patch", acquire)
    directory = evaluation.run(args)
    assert len(acquired) == 10 and len(set(acquired)) == 10
    assert loaders == [shared_model.MODEL_KWARGS, shared_model.MODEL_KWARGS]
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["outcomes"] == {"success": 8, "skipped": 1, "error": 1}
    assert manifest["csv_resolved_columns"] == {"diameter": "Diameter（m）"}
    assert manifest["category_definition_verified"] is True
    for field in ("run_id", "timestamp_utc", "category_mapping_version", "japan_filter",
                  "sampling_seed", "max_samples_per_category", "counts", "gsi_imagery_source",
                  "acquisition", "architecture", "preprocessing", "output_files", "known_limitations"):
        assert field in manifest
    assert manifest["saclaj_csv_sha256"] == data.sha256(args.saclaj_csv)
    assert manifest["base_checkpoint_sha256"] == data.sha256(args.base_model)
    assert manifest["fine_tuned_checkpoint_sha256"] == data.sha256(args.fine_tuned_model)
    assert str(tmp_path) not in json.dumps(manifest)
    assert "synthetic-private" not in json.dumps(manifest) + json.dumps(summary)
    assert (directory / ".gitignore").read_text() == "*\n"
    assert len(list((directory / "patches").glob("*.png"))) == 8
    with (directory / "site_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert next(row for row in rows if row["Category_ID"] == "9")["Category_detail"] == ""
    assert next(row for row in rows if row["Category_ID"] == "22")["Category_detail"] == "補足情報"
    assert "補足情報" not in json.dumps(summary, ensure_ascii=False)
    skipped = next(row for row in rows if row["subtype"] == "mixed")
    assert skipped["base_predicted_class"] == skipped["fine_tuned_predicted_class"] == ""
    assert summary["rice_agreement_percent"]["delta_pp"] == 100
    assert summary["agriculture_leakage_percent"]["delta_pp"] == 100


@pytest.mark.parametrize("diameter_column", ["Diameter（m）", "Diameter(m)"])
def test_offline_preflight_never_acquires_sites(tmp_path, monkeypatch, diameter_column):
    args, _ = setup_run(tmp_path, monkeypatch)
    write_csv(args.saclaj_csv, diameter_column=diameter_column, extra_empty_column=True)
    args.preflight = True
    def forbidden(*args):
        raise AssertionError("No imagery in preflight")
    monkeypatch.setattr(evaluation, "acquire_patch", forbidden)
    directory = evaluation.run(args)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "preflight_passed"
    assert manifest["csv_resolved_columns"] == {"diameter": diameter_column}
    assert not (directory / "site_results.csv").exists()


def test_all_acquisitions_fail_produces_null_metrics(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch)
    def missing(*args):
        raise evaluation.AcquisitionFailure("skipped", "imagery_unavailable")
    monkeypatch.setattr(evaluation, "acquire_patch", missing)
    directory = evaluation.run(args)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_pairs"
    assert summary["rice_agreement_percent"]["base"] is None
    assert summary["outcomes"]["skipped"] == 10


def test_private_paths_reject_git_checkouts(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="Git checkout"):
        evaluation.require_external(tmp_path / "data/private.csv")
    with pytest.raises(ValueError, match="outside this repository"):
        evaluation.require_external(Path(evaluation.__file__).parent / "results")


def test_model_failure_retains_local_error_and_failed_manifest(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "acquire_patch", lambda *args: np.zeros((512, 512, 3), dtype=np.uint8))
    original = evaluation.predict_pair
    calls = 0
    def fail_on_site(*values):
        nonlocal calls
        calls += 1
        if calls == 2:  # First call is the offline model check.
            raise RuntimeError("PRIVATE FAILURE DETAIL")
        return original(*values)
    monkeypatch.setattr(evaluation, "predict_pair", fail_on_site)
    with pytest.raises(RuntimeError):
        evaluation.run(args)
    directory = next(args.output_dir.iterdir())
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["outcomes"]["error"] == 1
    assert manifest["processed"] == 1 and manifest["unprocessed"] == 9
    assert summary["rice_agreement_percent"]["n"] == 0
    assert "PRIVATE FAILURE" not in json.dumps(manifest) + json.dumps(summary)


def test_preflight_rejects_empty_primary_or_leakage_population(tmp_path, monkeypatch):
    args, _ = setup_run(tmp_path, monkeypatch)
    write_csv(args.saclaj_csv, [sites_per_category()[0]])
    with pytest.raises(ValueError, match="non-Agriculture"):
        evaluation.run(args)
    write_csv(args.saclaj_csv, [sites_per_category()[-1]])
    with pytest.raises(ValueError, match="rice_paddy"):
        evaluation.run(args)
    assert not args.output_dir.exists()


def test_invalid_image_is_an_error(monkeypatch):
    monkeypatch.setattr(evaluation, "download_tile", lambda *args: Image.new("RGB", (1, 1)))
    with pytest.raises(evaluation.AcquisitionFailure, match="invalid_tile_dimensions"):
        evaluation.acquire_patch(sites_per_category()[0])
