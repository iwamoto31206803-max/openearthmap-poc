# Phase A / Phase B classification transition diagnostics

`compare_classification_rasters.py` は、同じグリッド上にある2枚の1-band OEM8分類GeoTIFFを
比較し、teacher追加前後のクラス遷移を数値化する。入力は class ID 0..8 でなければならず、
幅、高さ、CRS、affine transformが一致しない入力は拒否する。モデル重みや推論処理は不要である。

## 1比較の最小実行例

02（Paddy + Water）から03（Paddy + Water + Road）を比較する例:

```bash
python compare_classification_rasters.py \
  /local/predictions/02/classes.tif \
  /local/predictions/03/classes.tif \
  --label 02_to_03 \
  --output-dir /local/transition_diagnostics \
  --changed-geotiff
```

terminalには全pixel数、changed / unchangedの件数と構成比、および次の注目遷移を表示する。

- Pavement / Developed space -> Road
- Road -> Pavement / Developed space
- Buildings -> Tree
- Buildings -> Road
- Tree -> Road
- Grass / Rangeland -> Road
- Cropland / Agriculture -> Road
- Water -> Tree
- Cropland / Agriculture -> Tree
- Tree -> Buildings
- Background / Unlabelled -> 全クラス（unchangedの0 -> 0も含む）

出力先 `/local/transition_diagnostics/02_to_03/` には次を作成する。

- `summary.json`: 入力path、全81遷移、changed-only遷移、件数が1以上の
  `nonzero_changed_transitions`、主要遷移、AOI、changed / unchanged集計
- `transition_matrix.csv`: from/to 9 x 9のfull transition matrix（ゼロ件も含む）
- `changed_only.tif`: `from_id * 10 + to_id`。0はunchanged（オプション指定時のみ）

JSONとCSVの`percent_of_all`はAOI内の全pixelに対する構成比、
`percent_of_changed`はAOI内のchanged pixelに対する構成比である。changedが0件の場合、
`percent_of_changed`は0とする。入力GeoTIFFや生成物はローカルに保持し、Gitへ追加しない。

## AOI限定

row / colは0始まり、stopを含まないhalf-open rangeで指定する。

```bash
python compare_classification_rasters.py 02.tif 03.tif \
  --label 02_to_03_aoi \
  --rows 1000 1500 --cols 2000 2600 \
  --output-dir /local/transition_diagnostics
```

bboxはGeoTIFFのCRSで `LEFT BOTTOM RIGHT TOP` の順に指定する。pixelを部分的に切る曖昧さを
避けるため、bboxはpixel boundaryに一致させる。

```bash
python compare_classification_rasters.py 02.tif 03.tif \
  --label 02_to_03_bbox \
  --bbox 13900000 4200000 13901000 4201000 \
  --output-dir /local/transition_diagnostics
```

AOI指定時の`changed_only.tif`はcrop後のwidth / height / transformを持ち、元GeoTIFFのCRSを
維持する。

## 00 -> 01 -> 02 -> 03の一括比較

薄いラッパーは固定label `00_to_01`、`01_to_02`、`02_to_03` で3比較を作成する。

```bash
python compare_classification_progression.py \
  --stage-00 /local/predictions/00/classes.tif \
  --stage-01 /local/predictions/01/classes.tif \
  --stage-02 /local/predictions/02/classes.tif \
  --stage-03 /local/predictions/03/classes.tif \
  --output-dir /local/transition_diagnostics \
  --changed-geotiff
```

`--bbox`または`--rows`と`--cols`はラッパーでも使用でき、同じAOIが3比較すべてに適用される。
既存の非空label directoryへの誤上書きを防ぐため、再実行時は明示的に`--overwrite`を指定する。
