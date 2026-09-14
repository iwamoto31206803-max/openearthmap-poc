# GSI 部分教師データの準備

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
