# GSI paddy Phase A fine-tuning Pilot

目的は `Base model → GSI paddy partial-label fine-tuning → checkpoint` を
再現可能・監査可能な形で一度成立させることです。production modelの作成や、
精度向上の証明は目的としません。SACLAJによるBase vs Fine-tuned評価は別工程です。
このコマンドはSACLAJ、針葉樹、DCHM、筆ポリゴンを読み込みません。

## 入力と教師ラベル

既存 `prepare_gsi_labels.py` の出力を使います。既存データを再生成する必要はありません。
`org/` と `labels/` は同じ相対PNGパスで対応している必要があります。

```text
C:\OpenEarthMap_PoC\data\gsi\
├── raw\paddy_572\org\<source_image_id>.png
└── prepared\paddy_572\
    ├── labels\<source_image_id>.png
    ├── manifest.json
    └── audit.csv
```

サブディレクトリも保持します。source image IDは拡張子なしの相対パスです。
prepared manifestの `gsi_category="paddy"`、`oem_class_id=7`、
`ignore_index=255` と、画像数・FALSE数・教師画素数・総画素数を実ファイルと照合します。
ラベルは単一channel uint8 PNGとし、値7と255だけを許容します。
サイズ不一致、ペア欠落、大小文字だけ異なる重複パスはエラーです。

- **7:** OEM8 Cropland / Agricultureとして与えられたpositive教師画素。
- **255:** unknown / ignore。Backgroundでもnegativeでもありません。
- **FALSE画像:** 全画素255なので教師lossのsignalがありません。初回Pilotでは
  train/validationの両方から除外し、件数・理由・ID一覧を保存します。

提示された実データAuditは2,600画像、positive 156,376,781 / total 850,576,012画素
（約0.1838481）、FALSE 1,314画像、ラベル色RGB (0,255,255)です。
この値をコードへ固定しておらず、各runで検査します。このdatasetならusable 1,286枚、
既定splitはtrain 1,028枚 / validation 258枚です。

## Windows cmd + Python 3.11 venv

以下はリポジトリを `C:\OpenEarthMap_PoC\openearthmap-poc` に置いた例です。
既存venvを使う場合は作成・installを省略できます。Base重みのパスは実際の保存先に
合わせてください。重みやデータをダウンロード・生成する処理はありません。

```bat
cd /d C:\OpenEarthMap_PoC\openearthmap-poc
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

`py` launcherがない場合は、Python 3.11の `python.exe` のフルパスを使います。
会社PCでの最初の動作確認は、同じsplitと全positive画像を使う1 epochで実行します。

```bat
python -m src.training.train_gsi_paddy ^
  --org-dir C:\OpenEarthMap_PoC\data\gsi\raw\paddy_572\org ^
  --prepared-dir C:\OpenEarthMap_PoC\data\gsi\prepared\paddy_572 ^
  --base-model C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth ^
  --output-dir C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a ^
  --device cpu --batch-size 1 --num-threads 2 --epochs 1 --seed 42 --train-ratio 0.8
```

Pilot既定の3 epochsで実行するには `--epochs 3` とします。再実行ごとに一意なrun
subdirectoryを作り、過去runを上書きしません。CPUでは1 epochも長時間かかる可能性が
あるため、実データでの所要時間とメモリは会社PCで確認してください。

## architecture・preprocessing・freeze

`src/model.py` の共通loaderを推論と学習が使います。

```python
smp.Unet(
    encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3,
    classes=9, activation=None, decoder_attention_type="scse",
)
```

plain state_dict と `{"state_dict": ...}` を従来同様にstrict loadします。
Base重みと異なるarchitectureやキーの不整合を黙って無視しません。
uint8 RGB → float32 → /255.0 → CHWは既存推論と同じ共通関数です。
mean/std normalization、resize、crop、augmentationは行いません。

batch内の最大height/widthを32の倍数へ切り上げて右・下をpaddingします。
572×572なら576×576です。RGBは推論同様のedge replication、ラベルは255で埋め、
元の教師画素をすべて保持します。paddingはloss・metricに入りません。
サイズが異なる画像もbatch内paddingで扱えます。

encoderは全parameterの `requires_grad=False` に加えて、毎epochの `model.train()`
後にも `encoder.eval()` とし、BatchNormのrunning statisticsやdropoutも固定します。
decoderとsegmentation headだけをtrainableにし、optimizerにはtrainable parameter
だけを渡します。frozen/trainable parameter数を起動時とmanifestに記録します。

## 学習条件・再現性

| 項目 | 既定値 |
|---|---|
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8 |
| Epochs / batch size | 3 / 1 |
| Device | cpu (`auto`, `cuda` も選択可) |
| CPU threads / DataLoader workers | 2 / 0 |
| Seed / train ratio | 42 / 0.8 |
| Loss | CrossEntropyLoss(ignore_index=255, reduction="mean") |

CLIで `--learning-rate`, `--epochs`, `--batch-size`, `--device`, `--num-threads`,
`--seed`, `--train-ratio` を指定できます。CUDA未対応環境での明示的cuda指定はエラーです。
画像はオンデマンドで読み、データ全体やencoder featureをメモリに保持しません。

positive画像IDをsort後、ローカル `random.Random(seed)` でshuffleします。
train数は `floor(n * train_ratio)`（最低1枚・最大n−1枚）です。positiveが2枚未満なら
停止します。同じID集合・seed・ratioなら同じsplitです。Python/NumPy/PyTorchのseed、
DataLoader generatorを固定し、deterministic algorithmsを有効にします。
異なるhardwareやlibrary version間での数値の完全一致までは保証しません。

画像単位の位置情報がないため、**spatially independent splitではありません**。
隣接画像や類似場面が両splitに入る可能性があります。

## Lossとvalidationの解釈

class-7 **positive-only partial supervision**です。255はlossとmetricから除外し、
batchのlossは教師画素だけの平均、epochのlossは教師画素数で重み付けした平均です。
FALSE画像のnegative loss、class weighting、distillation、pseudo-labelingはありません。
all-ignore batchを誤って渡した場合はNaNを記録せずエラーにします。

train/validationそれぞれ、loss、labeled pixel count、class-7教師画素のprediction
agreement / recall、同画素上のclass-7平均確率を記録します。bestはvalidation loss最小
（同値なら先のepoch）です。これらはGSI weak/partial labelsへのagreementを監視する
training metricで、overall accuracyやaccuracy improvementではありません。
class 7への過剰予測や他クラスの劣化を検出できず、すべてclass 7と予測しても
agreementが高くなる点に注意してください。最終評価は別工程のSACLAJ evaluationです。

## 出力・推論互換性

```text
<output-dir>/<UTC timestamp + unique suffix>/
├── .gitignore                 # 任意の出力先でも内容をGit対象から除外
├── run_manifest.json
├── train_ids.json
├── validation_ids.json
├── excluded_ids.json          # 除外理由とID
├── dataset_inventory.json     # ID、サイズ、positive数、画像/ラベルSHA256
└── checkpoints/
    ├── epoch_001.pth           # 各epoch
    ├── best.pth               # validation loss最小
    └── final.pth              # 最終epoch
```

checkpointはCPU上のplain state_dictです。metadataは別JSONなので既存の
`src/predict_geotiff_tiled.py` の `build_model()` / `--model` でロードできます。
optimizer stateは保存せず、resume trainingは今回の範囲外です。

manifestにはrun ID/UTC日時、Baseパス/SHA256、architecture、preprocessing/padding、
GSI category/OEM8 class、prepared manifestのSHA256（内容そのものは転記しない）、
全件/usable/除外/train/validation数、分割方法/seed/ratio/IDファイル参照、全学習条件、
parameter数、device、Python/PyTorch/SMP/NumPy/Pillow versions、epochごとのmetrics、
best/final checkpoint、既知の制約を記録します。学習中の例外・中断時はstatus=failedと
exception型を記録し、完了済みepochを保持します。強制終了や電源断ではrunningのまま
残る場合があります。Base以外の入力絶対パスや、座標・地点情報は転記しません。

raw画像、prepared labels/audits、重み、checkpoints、training outputsはGitへ追加しない
でください。`data/`, `runs/`, `training_outputs/` と各runの `.gitignore` で除外します。
SACLAJデータもリポジトリ・外部環境へ追加しません。

## 実装検証と未確認事項

`python -m pytest -q` は一時ディレクトリで合成PNGと小さいdummy modelを使用します。
読み込み・正規化・ignore保持・572 padding・all-ignore除外・split・freeze/BN固定・
optimizer除外・ignore loss/gradient・manifest・checkpoint再読込を検証します。
実データやBase重みなしで実行でき、テスト用ファイルをGitに保存しません。

実データtraining、実Base重みと会社PCのSMP versionとの互換性、メモリ/所要時間、
学習後の他クラスへの影響は会社PCでの確認事項です。

Loss仕様: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
