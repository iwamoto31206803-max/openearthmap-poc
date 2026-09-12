# PoC Status — 2026-09-12

## 1. 目的

任意の対象範囲について国土地理院（GSI）の航空写真を取得し、
AIで土地被覆を推論して、技術者がQGISで確認・修正できるGISデータの
初期案を生成できるかを確認する。

本PoCの評価軸は、学術的精度だけではなく、
**人がゼロから判読・図化するより、AI出力を修正する方が実務上有利か**
という観点を含む。

## 2. 現在成立している処理

1. GSI seamlessphoto XYZ tile取得
2. RGB GeoTIFF化（EPSG:3857）
3. OpenEarthMap-SAR Optical pretrained weightによるtiled inference
4. class GeoTIFF / confidence GeoTIFF出力
5. raster sieveによる微小領域整理
6. GeoPackageへのpolygonize
7. QGISでの確認・編集
8. ポリゴン構造・面積・confidenceの集計

## 3. 使用中のモデル

- Weight: `RGB_Real_5_u-efficientnet-b4.pth`
- Architecture: U-Net / EfficientNet-B4 / scSE
- Input: RGB
- Output: 9 classes（Backgroundを含む）
- 現在の位置付け: **PoC / reference baseline**

利用条件については、企業内利用、fine-tuning、派生weight、
社内実行ファイルへの組込みの可否を提供者へ確認する。

## 4. 予備評価

### 良好・比較的良好
- Buildings
- Road
- Tree
- Grass

### 課題
- Water: 実河川は比較的良好。一方、樹木・構造物等の影をWaterと誤認する例がある。
- Cropland: 日本の農地でGrass / Pavement等との混同がみられる。
- Bareland: 現時点のサンプルでは十分な評価数がない。

## 5. Post-processing

Raw 3地区合計:
- Polygons: 3,173
- <5 m²: 1,774 polygons = 55.9%
- <5 m²領域の面積寄与: 約0.47%

5 m² sieve後:
- Polygons: 1,396
- Reduction: 約56.0%
- class変更画素: 各地区概ね0.6%未満

したがって、5 m² sieveを現時点の暫定標準値とする。

これはモデル誤認識を訂正する処理ではなく、
salt-and-pepper状の微小連結領域を整理するGIS後処理である。

## 6. 次フェーズ

- モデル非依存の評価地点・Ground Truth整備
- IoU / Precision / Recall / F1
- 「そのまま使用 / 軽微修正 / 大幅修正 / 手作業の方が速い」の実務評価
- 可能であれば手動図化時間 vs AI+修正時間
- SAR weightの利用条件確認
- ライセンス条件に応じて日本向けfine-tuningまたはrights-clean modelへ移行
