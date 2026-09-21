# GSI 年度別航空写真 + OEM-SAR validation GT 54 枚

## 目的とデータの扱い

日本向け評価用に、OEM-SAR validation 由来の既存 GT 54 枚と GSI 年度別航空写真
RGB を組にしたデータセットをローカルに構築する。54 地点の対応撮影年度は人手で同定済みで、
地域ごとに 2007、2017、2018、2019、2020、2021 年のいずれかへ固定する。
正式manifestは `manifests/val_gt_georef.csv` であり、dataset再構築用metadataとしてGit管理する。時期の違いに
よる評価誤差を避けるため、最新のシームレス航空写真ではなく manifest 指定年度を使う。
未対応年度や欠損 tile を別年度へフォールバックしない。

`src/build_gsi_val_gt54.py` は GT の CRS、transform、width、height を出力 RGB の基準にし、
年度別 XYZ tile の mosaic を bilinear 補間でその grid へ reproject する。GT はコピーと改名
だけを行い、画素、クラス ID、地理情報を変更せず、manual GT も新規作成しない。

## Dataset specifications

### Summary

| 項目 | 仕様 |
|---|---|
| Total items | 54 |
| Regions | 8 |
| GT type | manual OEM8 semantic segmentation labels |
| Label classes | OEM8 class IDs 0--8 |
| RGB source | GSI year-specific aerial imagery |
| RGB status | reconstructed / derived input |
| GT status | original manual label pixels preserved |
| Spatial reference | 対応する OEM-SAR validation SAR imagery から継承 |
| Output RGB grid | GT の CRS / transform / width / height と完全一致 |
| Full-run QC | 54 / 54 PASS |
| `alignment_ok` | 全54件で `True` |

GT54は **year-matched validation asset** かつ **full-scene manual-GT evaluation
asset** である。item-level mappingの正は
[`manifests/val_gt_georef.csv`](../manifests/val_gt_georef.csv)とし、54 ValAreaの一覧は
本文には重複記載しない。

### Region / year composition

正式manifestの `region` および `year` を集計した構成は次のとおりである。region名はmanifestの
既存値をそのまま記載する。

| Region | GSI year | Items |
|---|---:|---:|
| Tokyo_Shijuku | 2019 | 12 |
| Tokyo_Haneda | 2019 | 6 |
| Fukushima_Daiichi | 2018 | 6 |
| Kanazawa | 2007 | 6 |
| Toyohashi | 2020 | 6 |
| Moriguchi_Yodogawa | 2021 | 6 |
| Osaka_Port | 2017 | 6 |
| Mozu | 2021 | 6 |
| **Total** | **-** | **54** |

### Dataset interpretation / provenance

- RGBは公式に配布されたOEM-SAR validation optical RGBではない。各地点について人手で対応年度を
  確認したGSI年度別航空写真から再構成したderived inputである。
- GSI RGBは対応するGT gridへbilinearでreprojectする。一方、GTはdataset construction時に
  resampleせず、画素およびOEM8 class IDを変更しない。
- manual GTを新規作成したものではない。既存のOEM-SAR / DFC validation GTを地理参照して利用する。

Original Base checkpointのtraining provenanceは完全には確認できていない。GT54とBase training
dataのscene overlap、source-image overlap、およびOEM / OEM-SAR由来training dataとの関係は
いずれも未確認である。provenance auditが完了するまでは、GT54を **independent holdout**、
**independent test set**、または **independent evaluation set** として扱わない。

### File / grid specifications

各ValAreaの出力ペアには次の条件を適用し、full-run QCで検査する。

- RGBとGTは同一のwidth / height、CRS、affine transform、およびboundsを持つ。
- RGBは3-band raster、GTはOEM8 class IDs 0--8を値域とする1-band class rasterである。
- RGBのreprojectionにはbilinear resamplingを使う。
- GTはcopy / preserveとし、resamplingしない。

全54件に共通するtile sizeおよびpixel resolutionは、正式manifestには記録されていないため、
GT54全体の固定仕様としてはここで規定しない。

### Spatial overlap caveat

GT54には、同一region内で空間的に重複するValArea tileが含まれる。今後のevaluationでは単純な
54枚集計だけでなく、重複する可能性がある地理pixelを考慮する必要がある。de-duplicated
aggregation methodを含むevaluation protocolは、現時点では正式に確定していない。

## 実行

標準環境は Python 3.11 とする。まず代表 6 枚を実行する（Windows の `^` は行継続）。

```bat
python -m src.build_gsi_val_gt54 ^
  --manifest manifests\val_gt_georef.csv ^
  --gt-root C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef ^
  --output-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --ids ValArea_011 ValArea_008 ValArea_016 ValArea_038 ValArea_075 ValArea_110
```

代表6枚のpilot完了後、会社PC / local environmentでreal GSI tile downloadを伴う全件runを実施した。

```bat
python -m src.build_gsi_val_gt54 ^
  --manifest manifests\val_gt_georef.csv ^
  --gt-root C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef ^
  --output-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --all
```

出力は `rgb_images/`、`labels/`、`manifest.csv` に置く。正常な既存ペアは検証して再利用し、
再取得には `--overwrite` を付ける。timeout と retry は `--timeout`、`--retries` で指定できる。
失敗した地点も manifest に NG と理由を記録し、終了 code は 1 になる。partial file は削除する。

## Full-run verification

会社PCの実データ環境で54地点のfull runを完了し、次を確認した。

- **54 / 54 items PASS**
- generated `manifest.csv` の **`alignment_ok=True` を全54件**
- real GSI tile downloadにより、指定年度のRGBを各GT gridへreprojectできること

これにより、year-matched GSI RGB + georeferenced OEM8 GTのvalidation dataset構築を確認済みである。
生成したoutput dataset、download画像、GTコピー、QC結果はGit管理外とし、repositoryには正式manifest
`manifests/val_gt_georef.csv`だけを再現性metadataとして保持する。

## Training provenance上の注意

GT54は主要なfull-scene manual-GT evaluation assetとして使用する。ただし、Original Base
checkpointのtraining provenanceは完全には確認できておらず、GT54とBase training dataの
scene / source-image overlapも未確認である。Base checkpointの学習にOEM-SAR / OEM由来data、
今回の54地点、または同一source imageryが含まれていた可能性を現時点では排除できない。
provenance auditが完了するまでは、GT54をindependent holdoutとは扱わない。

## QC と制約

各ペアについて size、CRS、transform、bounds、band 数、OEM8 値域 (0--8)、RGB が空でない
ことを検査し、両ファイルの SHA256 と取得 URL template、zoom、tile 数を記録する。pixel grid
は GT transform を直接出力へ設定したうえで完全一致を要求する。出力 dataset、画像、GT、QC
成果物は Git 管理外である。

GSI航空写真の利用条件・出典表示、OEM-SAR GTの再配布条件、および両者を組み合わせたderived
datasetの再配布条件は未整理であり、公開・再配布前にそれぞれ別途確認する必要がある。また、
実データを使う pilot download はネットワークとローカル GT が必要なため、unit test では mock する。
