# OpenEarthMap PoC v0.1

## 目的

国土地理院（GSI）の航空写真から、OpenEarthMap-SAR系の学習済みRGBモデルを用いて土地被覆を推論し、
QGISで編集可能なGeoPackageまで生成するPoCです。

現時点では技術成立性の確認を目的としており、使用中の学習済みweight
`RGB_Real_5_u-efficientnet-b4.pth` の企業内利用・fine-tuning等の利用条件は確認中です。

## 現在の処理フロー

1. GSI「全国最新写真（シームレス）」からRGB GeoTIFFを取得
2. 512 pxタイル＋overlapで土地被覆を推論
3. Raw class GeoTIFFとconfidence GeoTIFFを保存
4. 5 m² sieveで微小連結領域を整理
5. 整理後class rasterをGeoPackageへpolygonize
6. ポリゴン数・面積等を解析
7. 実行条件と生成物をrun_manifest.jsonへ記録

## 推奨実行方法

```bat
python run_poc.py --lat 35.662 --lon 140.070 --name chiba_test --polygonize-raw
```

標準設定:
- GSI zoom: 18
- 取得範囲: 3 x 3 tiles
- 推論tile size: 512 px
- overlap: 128 px
- sieve: 5 m²
- connectivity: 8

## 生成物

各実行ごとに `runs/YYYYMMDD_HHMMSS_<name>/` が作られます。

```text
runs/
└─ YYYYMMDD_HHMMSS_<name>/
   ├─ 01_input/
   │  └─ gsi_rgb.tif
   ├─ 02_prediction/
   │  ├─ gsi_rgb_classes.tif
   │  └─ gsi_rgb_confidence.tif
   ├─ 03_postprocess/
   │  └─ gsi_rgb_classes_sieve_5m2.tif
   ├─ 04_vector/
   │  ├─ landcover_sieve.gpkg
   │  └─ landcover_raw.gpkg   # --polygonize-raw 指定時
   ├─ 05_analysis/
   │  ├─ overall_summary.csv
   │  ├─ threshold_summary.csv
   │  ├─ class_summary.csv
   │  └─ report.txt
   └─ metadata/
      └─ run_manifest.json
```

## 中間成果物を残す理由

- GSI取得結果そのものを確認できる
- モデルのRaw出力と後処理結果を分けて比較できる
- confidenceを後処理後のclass confidenceと誤解しない
- QGIS上でRaw / sieve後を比較できる
- 将来weightを変更した際に同一地点で再比較できる
- 上司報告・検証記録として処理条件を追跡できる

## 現時点のPoC所見

3地区でEnd-to-End処理を確認済み。

- Building / Road / Tree: 比較的良好
- Water: 河川は良好だが、樹木等の影をWaterと誤認する例あり
- Cropland: 日本の農地では不安定
- 5 m²未満の小ポリゴンはRawでは約56%を占めるが、面積寄与は約0.47%
- 5 m² sieveにより、変更面積0.6%未満程度でポリゴン数を約56%削減
- 出力GeoPackageはQGISで編集可能

## 注意事項

- confidenceは最大softmax確率であり、実測された正解率ではありません。
- sieveでclassを書き換えた画素にRaw confidenceを引き継いではいません。
- 現在のSAR系pretrained weightはPoC baselineとして使用中であり、本番業務利用は利用条件確認後に判断します。
