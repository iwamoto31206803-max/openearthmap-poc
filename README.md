# OpenEarthMap PoC v0.1

国土地理院（GSI）の航空写真から土地被覆を推論し、QGISで確認・編集できる
GeoTIFF / GeoPackage を生成する技術検証用PoCです。

> **Status:** Technical PoC / baseline.
> `RGB_Real_5_u-efficientnet-b4.pth` の企業内利用・fine-tuning等の利用条件は確認中です。

## Teacher progression diagnostics

`tools/evaluation/compare_teacher_progression.py` は、既存の単一 GSI RGB GeoTIFF に対して
Base → Paddy → Paddy+Water → Paddy+Water+Road の固定順で OEM8 推論差分を作ります。
これは accuracy の順位付けではなく、teacher 追加による変化の帰属を支援する診断です。

```bash
python -m tools.evaluation.compare_teacher_progression \
  --input /path/to/gsi_rgb.tif --name chiba01 \
  --base-model /path/to/base.pth \
  --paddy-model /path/to/paddy.pth \
  --water-model /path/to/paddy_water.pth \
  --road-model /path/to/paddy_water_road.pth \
  --focus-transition 85 --focus-transition 34
```

既定の出力先は `teacher_progression/<name>/` です。共有入力、4 stage、3つの隣接
transition、`progression_summary.json`、`manifest.json` を保存します。changed-only
raster は非変化画素を 0（QML では透明）にし、指定した transition の focus raster
も生成します。stage の GeoPackage には `--polygonize`、既存 run の置換には
`--overwrite` を明示してください。

## 推奨実行

リポジトリ直下から:

```bat
python run_poc.py --lat 35.662 --lon 140.070 --name chiba_test --polygonize-raw
```

1回の実行ごとに `runs/YYYYMMDD_HHMMSS_<name>/` を作成し、
入力画像、Raw推論、後処理、ベクタ、解析、実行条件を分けて保存します。

## Repository layout

```text
openearthmap-poc/
├─ run_poc.py          # 標準 End-to-End entry point
├─ src/                # production / reusable implementation
│  ├─ training/        # GSI fine-tuning
│  └─ evaluation/      # reusable evaluation code
├─ tools/
│  ├─ evaluation/      # analysis / comparison CLI tools
│  └─ datasets/        # dataset preparation utilities
├─ manifests/          # reproducibility metadata（実データではない）
├─ docs/               # design, experiment, and status records
├─ legacy/             # traceability のため保持する superseded code
└─ tests/              # automated tests
```

モデル重み、dataset、GeoTIFF、checkpoint、実行結果はこの構造へ追加せず、Git 管理外に
置きます。`manifests/` は dataset 再構築に必要な小さな metadata のみを管理します。

## Standard pipeline

```text
GSI seamlessphoto
   ↓
01_input/gsi_rgb.tif
   ↓ tiled inference
02_prediction/gsi_rgb_classes.tif
02_prediction/gsi_rgb_confidence.tif
   ↓ 5 m² sieve
03_postprocess/*_sieve_5m2.tif
   ↓ polygonize
04_vector/landcover_sieve.gpkg
   ↓
05_analysis/*.csv + report.txt

metadata/run_manifest.json
```

### 中間成果物を保存する理由

Raw推論と後処理を分離することで、モデル性能とGIS後処理の効果を混同せず確認できます。
また、将来weightを変更した際にも同一地点・同一条件で比較できます。

`confidence` は最大softmax確率であり、実測された正解率ではありません。
さらにsieveでclassが変更された画素について、Raw confidenceを後処理後のconfidenceとして扱いません。

## Current defaults

- GSI: 全国最新写真（シームレス）
- Zoom: 18
- Image area: 3 x 3 XYZ tiles
- Inference tile: 512 px
- Overlap: 128 px
- Sieve: 5 m² / 8-connectivity
- Model: `RGB_Real_5_u-efficientnet-b4.pth`
- Output CRS: EPSG:3857（GSI取得GeoTIFFを継承）

## Current PoC findings

3地区でEnd-to-End処理を確認しています。

- Building / Road / Tree: 比較的良好
- Water: 河川は比較的良好だが、影をWaterと誤認する例あり
- Cropland: 日本の農地では不安定
- Rawでは5 m²未満の小ポリゴンが全体の約56%だが、面積寄与は約0.47%
- 5 m² sieveでポリゴン数を約56%削減しつつ、変更画素は概ね0.6%未満
- GeoPackageをQGISで編集可能

現在の状況と次の優先事項は [Current Status](docs/CURRENT_STATUS.md) を参照してください。
次はmanual GTではなくGSI-only expansionを優先します。
`docs/POC_STATUS_20260912.md` は2026-09-12時点のhistorical snapshotとして保持しています。

GSI部分教師データの準備と監査については
`docs/GSI_TRAINING_DATA.md` を参照してください。

GSI Phase Aのunknown領域にBase teacherのKL preservationを加える任意モードは
[Base-Preservation Fine-tuning v0.2](docs/BASE_PRESERVATION_FINETUNING.md)を参照してください。
元Baseから再スタートする1 epoch Pilotで、既定のv0.1 positive-only動作は維持します。

GSI paddy Phase AのBase / fine-tuned checkpointを、SACLAJの独立した地点referenceで
paired比較するCLIは [SACLAJ Evaluation v0.1](docs/SACLAJ_EVALUATION.md) を参照してください。
Category_ID単位の確定mappingを同梱しています。templateへ配布READMEのSHA256を記入し、
まずoffline preflightを実行します。Category_detailは補足情報としてのみ扱います。
SACLAJの実CSV・座標・地点ID・地点別結果はリポジトリ外の会社PCローカル限定です。

manual GTなしで実施したteacher preparation → fine-tuning → SACLAJ development evaluationの
一区切りは、[Phase A Minimal End-to-End 実験記録](docs/PHASE_A_MINIMAL_E2E_SUMMARY.md)に
まとめています。これはproduction modelの成果報告ではありません。

Phase A v0.3のpreservation-only replay 1 epoch Pilotと固定SACLAJ development
sampleでの評価は完了しました。実行結果と制約は上記の
[Phase A実験記録](docs/PHASE_A_MINIMAL_E2E_SUMMARY.md)および
[Replay-Preservation v0.3](docs/REPLAY_PRESERVATION_V03.md)を参照してください。
v0.3はdevelopment evaluation上の候補であり、production readinessは未確認です。

## Model / license note

現行weightはOpenEarthMap-SARの公開pretrained modelをPoC baselineとして利用しています。
本番業務での利用、fine-tuning、社内ソフトウェアへの組込みについては、
モデル提供者への確認後に採否を判断します。

GSI航空写真および将来使用する教師データについても、それぞれの利用条件・出典表示を別途管理します。


## QGIS style

`run_poc.py` は成果物と同じbasenameの `.qml` を自動生成します。

- Land-cover class GeoTIFF: クラス別色、**不透明度50%**
- Land-cover GeoPackage: `class_id` によるカテゴリ分類、**不透明度50%**
- Confidence GeoTIFF: **0 = black / 1 = white の0–1グレースケール**

`.qml` は対応するTIF/GPKGと同じフォルダに置いたまま使用してください。
