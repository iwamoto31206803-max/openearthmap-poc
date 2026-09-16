# GSI 部分教師データの準備

準備済みのGSI paddy partial labelsを使用するPhase A fine-tuning Pilotは
[GSI_PHASE_A_TRAINING.md](GSI_PHASE_A_TRAINING.md) を参照してください。

W1 第1段階では、同じ相対パスにある `org/`（原画像）と `val/`（着色画像）の
PNGを対応付け、OEM8形式の部分教師ラベルと監査結果を生成します。

```text
dataset_root/
├── org/
│   └── area/1.png
└── val/
    └── area/1.png
```

リポジトリ直下で次のように実行します。クラスIDは `src/config.py` の既存OEM8
定義から、対象カテゴリに対応する値を明示してください。

```bash
python -m src.training.prepare_gsi_labels dataset_root prepared \
  --gsi-category tree --oem-class-id 5
```

既定では `val` の完全一致RGB `(255, 0, 0)` の画素だけが教師対象です。別の色を
使う場合は、例えば `--label-color 0,255,0` と指定します。出力先には、入力と同じ
相対パスの `labels/*.png`、画像別の `audit.csv`、全体集計の `manifest.json` が
作られます。入力PNGや生成結果をGitへ追加しないでください。

## 重要: 未着色画素は負例ではない

着色画素には指定した既存OEM8クラスIDを記録しますが、**未着色画素はBackground
や負例を意味しません**。ラベルPNGではすべて `ignore_index=255` とし、学習時の
損失計算対象から除外する必要があります。教師画素が0%の画像はFALSE画像として
監査CSVとマニフェストに数えられますが、その全画素も同様にignoreです。

処理は、欠落ペア、大小文字だけが異なる重複相対パス、画像サイズ不一致をエラーに
します。また、ラベル色では説明できない `org` と `val` の画素差を
`non_label_mismatch_count` として監査します。撮影時期・地区を別途与えないこの段階
では、撮影日は空欄、精度は `unknown`、地区は空欄です。

## org / val 着色方式の診断

教師生成ロジックを変更する前に、読み取り専用の診断コマンドで差分方式を確認できます。
Windows の実データに対する実行例です（出力先もGit管理外にしてください）。

```bat
python -m src.training.diagnose_gsi_overlay ^
  C:\OpenEarthMap_PoC\data\gsi\raw\paddy_572 ^
  C:\OpenEarthMap_PoC\data\gsi\diagnostics\paddy_572
```

全ペアを1枚ずつ読み、`per_image.csv` に画像ID、サイズ、完全一致・非一致画素数と比率を、
`summary.json` にRGB各channelの符号付き `val - org` histogram、丸めたRGB色差の大きさ、
頻出する差分RGB、Otsu閾値、非一致画素に対するalpha-blend最小二乗推定を記録します。
既定の代表画像は `1`、`1300`、`2600` で、`--representative` で変更できます。入力パス、
元画像、画素値の位置や座標は出力しません。

判断時には次の順で確認します。

1. 固定色は、実データに完全一致するsentinel色が確認できた場合だけ使用する。
2. 単純非一致は、非教師領域が完全一致し、表示処理等による微差がない場合だけ候補にする。
3. RGB差分閾値は、色差histogramに画像間で安定した谷がある場合に候補にする。
4. alpha-blend方式は、推定alphaとoverlay RGBが物理範囲内で、残差RMSEが十分小さい場合に
   候補にする。

診断には正解maskがないため、`summary.json` だけで方式を自動確定してはいけません。
候補方式のfalse positive / false negativeを、複数画像の目視maskで検証してから教師生成を
変更してください。診断出力も実行結果としてGitへ追加しないでください。
