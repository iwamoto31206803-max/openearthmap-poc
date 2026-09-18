# Phase A v0.3: all-ignore preservation-only replay

v0.2のpositive-bearing画像へのadaptationを維持したまま、これまで除外していた
all-ignore / FALSE画像をBase分布の保持に使うPilotです。CLIでは
`--training-mode base_preservation_replay`を指定します。v0.1 / v0.2の動作は維持します。
**v0.3の精度改善はまだ確認していません。** SACLAJはdevelopment evaluationであり、
final acceptanceではありません。このtraining CLIはSACLAJを読みません。

## 背景: Water degradation

[既存の実験記録](PHASE_A_MINIMAL_E2E_SUMMARY.md)では、Water → Waterのagreementは
Base 81%に対してv0.2 34%でした。利用者が報告した追加診断では、Base-correct 81地点のうち
47地点でWaterを失い、そのうち41地点（87.2%）がAgricultureへ移りました。
GSI positive 156,376,781画素のうち33,129,469画素（21.1857%）はBaseのargmaxがWaterで、
conflictは一部の水田画像へ集中していました。難しい湛水水田もAgricultureとして学習する
ため、こうしたpositive画像は除外しません。

all-ignore 1,314画像ではBase argmax Waterの画素割合が約11.1249%でした。ただしこれは
正解ラベルではありません。**all-ignore画像はWater教師でも、Agriculture-negative教師でも
ありません。** 全9クラスのsoft distributionを保存するためだけに使用します。これらの
診断値は利用者報告であり、このPRで実データの再評価を行った値ではありません。

## 原Base・比較条件

Studentと別インスタンスのTeacherを、同じ**original Base**からstrict loadします。
**v0.2 checkpointから継続学習しません。** `--base-model`に元の
`RGB_Real_5_u-efficientnet-b4.pth`を指定してください。preflight、smoke、Pilotはそれぞれ
元Baseから開始する独立runです。resume機能はありません。

BaseのSHA256をロード前後に照合し、student/teacherの全state（buffer含む）の初期一致を
検査します。`--base-sha256`にはv0.2 manifestの`base_model_sha256`を指定すると、
既知の元Baseとの同一性を強制できます。配布元の既知hashを埋め込んでいないため、
任意のファイルを「original Base」と自動認証することはできません。

- positive split: v0.2の`split_samples`をそのまま使用。seed=42、ratio=0.8をv0.3では検査。
- 1,286 positive → train 1,028 / validation 258。
- 1,314 all-ignore → train 1,051 / validation 263。同じsort/local shuffle/floor helperで導出。
- replayのshuffleは独立したローカルRNG。4集合のID重複・混入を検査。
- λ=1、T=1、α=1、AdamW lr=1e-4、weight_decay=0.01、batch=1、1 epochを既定。
- architecture、RGB /255、augmentationなし、encoder parameter/BN/dropout固定はv0.2と同じ。
- decoder/headのみ学習。Teacherは全parameter固定、eval/no_grad、optimizer外。
- RGB=edge、label=255、32の倍数へ右下padding。同じ`collate_preservation`で実画像maskを作成。

このPRではλ/T/αの探索は行いません。v0.3はλ>0、T>0、α>=0の有限値を要求します。
v0.1/v0.2で従来許されていたλ=0は維持します。

## 厳密なlossとbatching

P = positive-bearing batch内の実画像のlabel 7、U = 同batch内の実画像のlabel 255、
R = replay batch内の全実画像画素。synthetic paddingはすべて対象外です。
teacher分布p_T=softmax(Base/T)、student分布q_T=softmax(Student/T)として:

```text
CE_P = mean_P CE(Student, 7)
K_U  = T² × mean_U sum_c p_T(c) log(p_T(c) / q_T(c))
K_R  = T² × mean_R sum_c p_T(c) log(p_T(c) / q_T(c))

L_positive = CE_P + λ K_U          # v0.2の関数をそのまま使用
L_replay   = λ K_R                # CEは一切計算しない
L_total    = L_positive + α L_replay
```

CE_P、K_U、K_Rはそれぞれのeligible画素だけで独立に平均してから合成します。
KLは`F.kl_div(log_softmax(student/T), softmax(teacher.detach()/T), reduction="none")`
のclass軸をsumし、対象画素をmeanしてT²を掛けます。PにKL、U/RにCEはありません。
Uが空ならK_U=0。positiveが空、replayにlabel 7がある、Rが空ならエラーです。

1 logical stepはpositive 1 batch + replay 1 batch → 1 total backward → 1 optimizer.step。
epoch長はpositive training loaderで決まり、optimizer step数はv0.2と同じです。
replayが先に尽きた場合は同loaderの新しいiteratorを作り、seed固定generatorの続きで
決定的にreshuffleします。画像tensorを`itertools.cycle`等でキャッシュしません。
replayが長い場合はpositive epoch終了時に止めるため、毎epoch全replay画像を消化する
保証はありません。既定batch=1では1,028 replay batches / epochです。

validationはpositive、replayそれぞれの全validation画像を1回ずつ評価します。
replay validationはtrainingに入りません。preflight/smokeだけ各splitを1 batchに制限します。
α=0でもreplay KLを監視しますが、replay student forwardはeval/no_gradにして
decoderのBN/dropout状態を変えず、v0.2の更新と乱数列を維持します。

## Metrics / provenance

training / validationとも次を保存します。lossの成分値はλ/αを掛ける前です。

| field | 意味 |
|---|---|
| positive_ce_loss | CE_P |
| unknown_preservation_loss | K_U（T²込み） |
| replay_preservation_loss | K_R（T²込み） |
| total_loss / loss | CE_P + λ K_U + α λ K_R |
| weighted_replay_preservation_loss | α λ K_R |
| labeled_pixel_count | P画素数 |
| unknown_preservation_pixel_count | U画素数 |
| replay_preservation_pixel_count | R画素数（再周回分も含む） |
| batch_count / replay_batch_count | 処理したpositive / replay batch数 |

v0.2のpositive CE/KL field、class-7 agreement、mean class-7 probability、unknown Base
agreementも維持します。epoch集計はP/U/Rそれぞれの画素数で加重平均してから合成します。
best checkpointはvalidation combined loss最小（同値は最初）。training metricは更新前の
各forwardに基づきます。これらはaccuracyやWater保持の直接評価ではありません。

schema_version=3、model_version=`gsi_phase_a_v0.3`。run ID/status、Base path/SHA256、
student/teacher初期SHA256、best checkpoint SHA256、seed/λ/T/α、optimizer/LR/epoch数、
positive/replay train/validation件数、Python/torch/SMP/NumPy/Pillow versionsを記録します。
以下の4ファイルと、それぞれのSHA256をmanifestへ保存します:

```text
train_ids.json                  # positive, v0.2と同じ形式
validation_ids.json             # positive
replay_train_ids.json
replay_validation_ids.json
```

`git_commit_sha`は実行時にこのソースを含むrepositoryの`git rev-parse HEAD`を取得し、
gitなし・失敗・timeout時はnullとします。実行コードを識別できるよう、会社PCではcommit済みの
clean checkoutを使用してください。過去v0.1/v0.2の未記録git SHAは補完しません。
`excluded_ids.json` / `excluded_all_ignore_images`は従来互換のため**positive datasetからの
除外**を表し、v0.3での用途は`all_ignore_usage`に記録します。preflightではbest hashはnull。

## 会社PC: Windows cmd.exe / Python 3.11

実際の保存先にpathを合わせ、v0.2と同じvenvを使用してください。
`BASE_SHA`には**v0.2 run_manifest.jsonのbase_model_sha256**を転記します。

```bat
cd /d C:\OpenEarthMap_PoC\openearthmap-poc
call .venv\Scripts\activate.bat
set "ORG=C:\OpenEarthMap_PoC\data\gsi\raw\paddy_572\org"
set "PREPARED=C:\OpenEarthMap_PoC\data\gsi\prepared\paddy_572"
set "BASE=C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
set "BASE_SHA=ここをv0.2 manifestのbase_model_sha256の64桁へ置換"
set "RESULTS=C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a_v03"

python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" --base-sha256 "%BASE_SHA%" ^
  --output-dir "%RESULTS%\preflight" --training-mode base_preservation_replay ^
  --lambda-preserve 1 --temperature 1 --alpha-replay 1 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 --seed 42 --train-ratio 0.8 --preflight

python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" --base-sha256 "%BASE_SHA%" ^
  --output-dir "%RESULTS%\smoke" --training-mode base_preservation_replay ^
  --lambda-preserve 1 --temperature 1 --alpha-replay 1 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 --seed 42 --train-ratio 0.8 --smoke-test

python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" --base-sha256 "%BASE_SHA%" ^
  --output-dir "%RESULTS%\pilot" --training-mode base_preservation_replay ^
  --lambda-preserve 1 --temperature 1 --alpha-replay 1 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 --seed 42 --train-ratio 0.8
```

preflightは全PNGのpair/size/label/hash/count監査（positive=1,286 / all-ignore=1,314）、
各splitのdisjoint検査、Baseのstrict load/同一性、teacherのoptimizer除外、encoder固定を
検査します。positiveとreplay各train/validationの1 batchをeval/no_gradでforwardし、
finite lossを確認します。checkpointやoptimizer更新はありません。

smokeは全audit後にpositive 1 batch + replay 1 batchで1回更新します。finite loss、backward/
step成功、teacher/encoderのparameterとbuffer不変、decoder/head更新、teacher gradientなしを
検査します。replay専用lossはCEを持たず、全label=255をassertします。α>0ではreplay logits
に対するgradientを検査し、paddingが0であることを確認します（α=0ではこの検査値はnull）。
両validationの各1 batchも評価し、`smoke_checks`を記録します。smoke checkpointはPilotの
代用やresume元にしません。成功statusは`preflight_passed` / `smoke_test_completed` / `completed`。

v0.3既定出力は`training_outputs/gsi_phase_a_v03/`で、毎回一意のrun directoryを作ります。
v0.2出力は上書きしません。重み・データ・実行結果・SACLAJ CSV/座標/地点別結果はGitに
追加しません。各run内の`.gitignore`も維持します。

## 開発検証と限界

```bat
python -m pytest -q
python -m compileall -q run_poc.py src tests
```

合成PNG/logits/TinyModelでsplit同一性、replay mask/KL/CEなし、α=0、1-step構造、
teacher/encoder不変、manifest/hash/git unavailable、preflight/smokeを検証します。
会社PCの実Base/実GSIによるpreflight・smoke・Pilot成功、RAM・所要時間、Water改善は別確認です。
2つのstudent forwardの計算graphを保持するため、v0.2よりメモリと計算量が増えます。
Teacherの誤りも保持し得ます。画像単位splitはspatial independenceを保証しません。
SACLAJには時期差・point referenceとしての限界があり、今回のPRで評価は実行しません。

## Company-PC execution result

### Preflight

- **PASS**
- images: 2,600（positive-bearing 1,286 / all-ignore 1,314）
- positive train / validation: 1,028 / 258
- original Base SHA256: `852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c`
- encoder: frozen
- decoder + segmentation head: trainable
- teacher: frozen / eval / no-grad / optimizer excluded
- training performed: no

### Smoke

- **PASS**
- one positive batch + one replay batch
- backward / `optimizer.step()` succeeded; runtime errorなし
- smoke-onlyのaccuracy値はperformance evidenceではない

### 1-epoch Pilot

- status: **completed**
- run ID: `20260917T160534_801449Z_8c785b89`
- git commit SHA: `23b1862cdda6ba30dedde9534e349f8c8c564d79`
- Base SHA256: `852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c`
- best checkpoint SHA256: `e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172`
- positive train / validation: 1,028 / 258
- replay train / validation: 1,051 / 263
- `alpha_replay`: 1.0
- best epoch: 1
- best validation loss: 1.4663871948093399

| Split | Total loss | Positive CE | Unknown preservation | Replay preservation |
|---|---:|---:|---:|---:|
| Training | 3.0154720784697524 | 1.0549066109461889 | 0.8355242133963964 | 1.1250412541271675 |
| Validation | 1.4663871948093399 | 0.5980995695092703 | 0.4140308306615627 | 0.4542567946385068 |

| Validation metric | Value |
|---|---:|
| labeled pixels | 30,789,226 |
| unknown-preservation pixels | 53,624,246 |
| replay-preservation pixels | 86,049,392 |
| positive batches | 258 |
| replay batches | 263 |
| class-7 labeled-pixel agreement recall | 0.862370460368182 |
| class-7 mean probability on labeled pixels | 0.6076116140387008 |
| unknown student/Base argmax agreement | 0.736145418249797 |

### SACLAJ development evaluation aggregate

評価run `20260918T000738_565461Z_3d6b2f0b`は固定sample 1,000点
（10 subtype × 100）全てで成功し、skipped 0 / error 0だった。

| Metric | Base | v0.3 |
|---|---:|---:|
| Rice → Agriculture | 27% | 87% |
| Other crop → Agriculture | 24% | 61% |
| Agriculture leakage | 3.375% | 2.625% |
| Broadleaf → Tree | 89% | 91% |
| Needleleaf → Tree | 83% | 87% |
| Mixed → Tree | 87% | 88% |
| Water → Water | 81% | 72% |
| Needleleaf evergreen → Tree | 88% | 95% |
| Broadleaf evergreen → Tree | 88% | 90% |
| Needleleaf deciduous → Tree | 85% | 89% |
| Broadleaf deciduous → Tree | 79% | 81% |

Selected probability aggregatesは次のとおりである。これらはcalibrated accuracyではない。

| Probability aggregate | Base | v0.3 |
|---|---:|---:|
| Rice Agriculture | 0.25526936685715557 | 0.6225228072702884 |
| Other crop Agriculture | 0.23120718797888912 | 0.43006271876161917 |
| Water expected class | 0.7598597421547311 | 0.5506678780331277 |
| Water Agriculture | 0.008055887053861852 | 0.18305382947072757 |
| Broadleaf Tree | 0.8581725124397781 | 0.7337022725865245 |
| Needleleaf Tree | 0.7906729496899061 | 0.6593377662077546 |
| Mixed Tree | 0.838368598338493 | 0.6869575675623492 |

### v0.2 → v0.3 evaluation input identity

ローカルread-only比較は **PASS** だった。sampled / successful site setとsubtype countsは一致し、
共通の成功1,000点のpatch SHA256は1,000件全て一致した（different 0 / missing 0）。
Base argmax、Base expected probability、Base Agriculture probabilityの不一致もそれぞれ0である。

v0.2とv0.3のSACLAJ development evaluationsは同じsampled sitesとbyte-identicalな保存済み
GSI patchesを使用した。そのため、観測されたv0.2–v0.3指標差は同じ評価入力上の
model-output differencesとして扱える。

## Interpretation and limitations

v0.3は固定SACLAJ development sample上で、v0.2よりWater retentionを多く回復させつつ、
実質的なAgriculture adaptationと低いleakageを示した。このより好ましい
adaptation-preservation trade-offによりv0.3を現時点で最も有力なPhase A学習方式
candidateとする。

ただし、SACLAJはv0.1 / v0.2の結果が後続設計に影響したdevelopment evaluationであり、
unseen holdoutではない。point reference、sampling、空間重複、撮影時期差の限界も残る。
本結果はproduction accuracyまたはproduction readinessの主張ではない。最終acceptanceと
モデル選定には、spatially/unseen evaluationとmanual GTまたは同等の独立データが必要である。

SACLAJ CSV、座標、ID、地点別結果、patch、training output、checkpointはGitに保存せず、
会社PCローカル限定とする。本文書はaggregate値と実行provenanceのみを記録する。
