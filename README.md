# OpenEarthMap PoC v0.1

国土地理院（GSI）の航空写真から土地被覆を推論し、QGISで確認・編集できる
GeoTIFF / GeoPackage を生成する技術検証用PoCです。

> **Status:** Technical PoC / baseline.
> `RGB_Real_5_u-efficientnet-b4.pth` の企業内利用・fine-tuning等の利用条件は確認中です。

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
├─ run_poc.py                 # 標準End-to-End実行入口
├─ requirements.txt
├─ README.md
│
├─ src/                       # 現在の本線コード
│  ├─ config.py               # クラス定義・PoC既定値
│  ├─ download_gsi_geotiff.py
│  ├─ predict_geotiff_tiled.py
│  ├─ sieve_landcover.py
│  ├─ polygonize_landcover.py
│  ├─ analyze_landcover_gpkg.py
│  └─ qgis_styles.py
│
├─ legacy/                    # 初期技術検証用。標準処理では使用しない
│  ├─ check_rgb_model.py
│  ├─ download_gsi_image.py
│  ├─ predict_rgb.py
│  └─ predict_rgb_tiled.py
│
├─ docs/
│  └─ POC_STATUS_20260912.md
└─ examples/
   └─ README.md
```

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

詳細は `docs/POC_STATUS_20260912.md` を参照してください。

GSI部分教師データの準備と監査については
`docs/GSI_TRAINING_DATA.md` を参照してください。

GSI paddy Phase AのBase / fine-tuned checkpointを、SACLAJの独立した地点referenceで
paired比較するCLIは [SACLAJ Evaluation v0.1](docs/SACLAJ_EVALUATION.md) を参照してください。
Category_ID単位の確定mappingを同梱しています。templateへ配布READMEのSHA256を記入し、
まずoffline preflightを実行します。Category_detailは補足情報としてのみ扱います。
SACLAJの実CSV・座標・地点ID・地点別結果はリポジトリ外の会社PCローカル限定です。

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
