# GSI 年度別航空写真 + OEM-SAR validation GT 54 枚

## 目的とデータの扱い

日本向け評価用に、OEM-SAR validation 由来の既存 GT 54 枚と GSI 年度別航空写真
RGB を組にしたデータセットをローカルに構築する。54 地点の対応撮影年度は人手で同定済みで、
地域ごとに 2007、2017、2018、2019、2020、2021 年のいずれかへ固定する。時期の違いに
よる評価誤差を避けるため、最新のシームレス航空写真ではなく manifest 指定年度を使う。
未対応年度や欠損 tile を別年度へフォールバックしない。

`src/build_gsi_val_gt54.py` は GT の CRS、transform、width、height を出力 RGB の基準にし、
年度別 XYZ tile の mosaic を bilinear 補間でその grid へ reproject する。GT はコピーと改名
だけを行い、画素、クラス ID、地理情報を変更せず、manual GT も新規作成しない。

## 実行

標準環境は Python 3.11 とする。まず代表 6 枚を実行する（Windows の `^` は行継続）。

```bat
python -m src.build_gsi_val_gt54 ^
  --manifest C:\OpenEarthMap_PoC\oemsar_data\manifests\val_gt_georef.csv ^
  --gt-root C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef ^
  --output-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --ids ValArea_011 ValArea_008 ValArea_016 ValArea_038 ValArea_075 ValArea_110
```

pilot の全行が `PASS` で `manifest.csv` の `alignment_ok=True` になった後に全件を実行する。

```bat
python -m src.build_gsi_val_gt54 ^
  --manifest C:\OpenEarthMap_PoC\oemsar_data\manifests\val_gt_georef.csv ^
  --gt-root C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef ^
  --output-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --all
```

出力は `rgb_images/`、`labels/`、`manifest.csv` に置く。正常な既存ペアは検証して再利用し、
再取得には `--overwrite` を付ける。timeout と retry は `--timeout`、`--retries` で指定できる。
失敗した地点も manifest に NG と理由を記録し、終了 code は 1 になる。partial file は削除する。

## QC と制約

各ペアについて size、CRS、transform、bounds、band 数、OEM8 値域 (0--8)、RGB が空でない
ことを検査し、両ファイルの SHA256 と取得 URL template、zoom、tile 数を記録する。pixel grid
は GT transform を直接出力へ設定したうえで完全一致を要求する。出力 dataset、画像、GT、QC
成果物は Git 管理外である。

GSI tile の利用条件・出典表示、および OEM-SAR GT と組み合わせたデータセットの再配布を含む
ライセンス整理は未解決事項であり、公開・配布前に別途確認する必要がある。また、実データを
使う pilot download はネットワークとローカル GT が必要なため、unit test では mock する。
