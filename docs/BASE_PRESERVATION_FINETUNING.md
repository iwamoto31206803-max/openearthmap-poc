# Base-Preservation Fine-tuning v0.2

GSI paddy positiveへのadaptationを維持しつつ、unknown領域でBaseの9-class出力分布を
保存するPilotです。既存の[Phase A v0.1](GSI_PHASE_A_TRAINING.md)から明示的に
`--training-mode base_preservation`で選択します。指定しない場合は従来のpositive-only。
v0.1の既定値は変更せず、epochs未指定時だけmodeごとにv0.1=3、v0.2=1とします。

## 背景と比較条件

利用者が報告した独立SACLAJ評価では、Base → v0.1でRice agreement 27% → 99%、
Agriculture leakage 3.375% → 39.25%、Broadleaf → Tree 89% → 13%、
Needleleaf → Tree 83% → 7%、Mixed → Tree 87% → 17%、Water → Water 81% → 6%。
これらは利用者報告であり、この開発環境で実データを再評価した値ではありません。
positive-only supervisionによるclass 7側へのoutput driftを抑えられるかを検証します。

**v0.1 checkpointから継続しません。** `--base-model`には元の
`RGB_Real_5_u-efficientnet-b4.pth`を指定し、同じファイルからstudentとteacherを別々に
strict loadします。`best.pth`/`final.pth`へ置き換えないでください。
resume機能もteacher専用checkpoint指定もありません。

同じcheckpointを両方に使ったことはSHA256で記録し、ロード前後の変更も検出します。
元Baseの既知SHA256があれば`--base-sha256`で照合できます。ただし配布元Baseのhashは
コードに埋め込んでいません。ファイル名・同一hashだけで「元のBase」とは認証できません。
v0.1 run manifestの`base_model_sha256`と照合すると、両runの初期重みを比較できます。

同じprepared paddy_572、positiveが1画素以上ある画像のみ、同じdeterministic split
（seed=42、train ratio=0.8）、batch=1、AdamW lr=1e-4、weight_decay=0.01、
encoder固定、decoder/headのみ更新、augmentationなし、CPU対応を維持します。
all-ignore画像はpreservation signalがあってもv0.1との比較のため引き続き除外します。

## Lossの厳密な定義

元画像内の画素に限り、P={label=7}、U={label=255}とします。
255はBackground/negativeではありません。クラスは既存ID 0–8の9出力すべてを使います。
student logitsをs、Base teacher logitsをb、temperatureをTとして、

```text
p_T(i,c) = softmax(b(i,:)/T)[c]       # detached teacher
q_T(i,c) = softmax(s(i,:)/T)[c]       # trainable student

L_GSI = (1/|P|) Σ_{i∈P} -log softmax(s(i,:))[7]
L_preserve = T² × (1/|U|) Σ_{i∈U} Σ_{c=0..8} p_T(i,c) log(p_T(i,c)/q_T(i,c))
L_total = L_GSI + lambda_preserve × L_preserve
```

KLの方向は **KL(Base teacher || student)**。
PyTorchでは`F.kl_div(input=log_softmax(student/T), target=softmax(teacher.detach()/T),
reduction="none", log_target=False)`のclass軸をsumしてunknown画素をmeanにします。
`reduction="mean"`によるclass数分の追加除算や、`batchmean`によるbatch数除算はしません。
CEとKLは**それぞれの画素数で独立に平均してから**加算します。
positiveへのKL、unknownへのGSI CEは適用しません。CEにはtemperatureを掛けません。

既定λ=1.0、T=1.0。将来のT変更でもstandard distillationのT²スケーリングを維持しますが、
今回のPilotはT=1.0固定です。λは有限非負、Tは有限正数を要求します。
λ=0ならpreservationの勾配寄与は0で、v0.1のCEと同じ目的関数です。
その場合もteacher forwardとKL監視は行うので実行コストはv0.1より増えます。
teacher構築時のCPU RNGを復元し、追加モデル初期化によるstudentの乱数列変化を防ぎます。
演算の形が違うため浮動小数点のbit単位一致は保証しません。

|P|=0のbatchは従来どおりエラー。|U|=0はpreservation contribution=0とし、
unknown KL/agreementの監視値はnull（観測なし）にします。

## Paddingとmodel lifecycle

`src/model.py`の共通architecture / loader / preprocessingは変更しません。
uint8 RGB → float32 /255 → CHW、右・下を32の倍数へRGB=edge / label=255でpadding。
572×572なら576×576です。追加の`image_mask`で元の矩形範囲だけを保持し、
**synthetic paddingはCE・KL・metricすべてから除外**します。
これはunknownの選別ではなく、実在しない追加画素を除く処理です。
従来の`collate_padded`の戻り値・動作は維持します。

- Student: 元Baseからロード → encoder parameter固定 → training時decoder/headをtrain。
  `model.train()`の後もencoder.eval()でencoder BN/dropoutを固定。
- Teacher: 同じBaseを別インスタンスとしてロード → 全parameter requires_grad=False → eval。
  各epoch/validation呼出しでevalを再設定し、forwardは常にtorch.no_grad()。
  全moduleのBN statistics/dropoutを固定。teacherはoptimizerに入りません。
- 同じ前処理済みRGB tensorを両モデルへ渡します。teacherはEMA更新・再学習しません。
- 保存対象はstudentだけ。epoch / best / finalすべてplain CPU state_dictで、
  既存inference / SACLAJ loaderでstrict loadできます。optimizer/teacherは保存しません。

## Training / validation metricsとmanifest

両splitで同じobjectiveを計算します。epoch集計ではCEをpositive数で、preservationを
unknown数で個別に加重平均し、epoch CE + λ × epoch preservationを`loss`にします。
batch total lossの単純平均や、totalをpositive数で加重平均する方式ではありません。
best checkpointはこのvalidation combined loss最小、同値なら先のepochです。
training値は各更新前のforward出力に基づきます。

| JSON metric | 意味 |
|---|---|
| loss | combined total loss |
| gsi_positive_ce_loss | positive CE平均 |
| base_preservation_kl_loss | unknown KL平均 × T² |
| weighted_preservation_loss | λ × preservation loss |
| labeled_pixel_count | positive画素数 |
| unknown_preservation_pixel_count | 元画像内unknown画素数 |
| class_7_labeled_pixel_agreement_recall | positive上のclass-7 argmax一致割合 |
| class_7_mean_probability_on_labeled_pixels | positive上のT=1 class-7確率平均 |
| unknown_mean_kl_divergence | unknown上のKL(p_T‖q_T)平均、T²を含まない |
| unknown_student_base_argmax_agreement | unknown上のstudentとBaseのargmax一致割合 |
| batch_count | 実際に処理したbatch数 |

これらはoverall accuracyではありません。Teacher一致は正解一致を意味しません。
T=1のPilotでは`base_preservation_kl_loss`と`unknown_mean_kl_divergence`は同値です。

v0.2 manifestはschema_version=2。従来のdataset audit、split、versions、checkpoint
参照を維持し、training_mode、teacher / initialization checkpoint SHA256、同一Baseの宣言、
teacher frozen/eval/no_grad/optimizer除外/BN固定、loss定義、KL方向、PyTorch入力形式、T、λ、
両normalization、両mask、epoch aggregation、各epochのtrain/validation component losses、
known limitationsを記録します。`expected_base_sha256`は任意の照合用hashです。

## 会社PC: preflight → 1-step smoke → 1 epoch Pilot

Windows **cmd.exe**、Python 3.11の既存Phase A venvを使用します。
以下のpathは例なので実際の保存先へ変更してください。

```bat
cd /d C:\OpenEarthMap_PoC\openearthmap-poc
call .venv\Scripts\activate.bat
set "ORG=C:\OpenEarthMap_PoC\data\gsi\raw\paddy_572\org"
set "PREPARED=C:\OpenEarthMap_PoC\data\gsi\prepared\paddy_572"
set "BASE=C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
set "RESULTS=C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a_v02"
certutil -hashfile "%BASE%" SHA256
```

上のhashがv0.1の`base_model_sha256`と一致することを確認します。
任意で以下3コマンドに`--base-sha256 実際の64桁hash`を追加すると一致を強制できます。

```bat
python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" ^
  --output-dir "%RESULTS%\preflight" --training-mode base_preservation ^
  --lambda-preserve 1.0 --temperature 1.0 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 ^
  --seed 42 --train-ratio 0.8 --preflight

python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" ^
  --output-dir "%RESULTS%\smoke" --training-mode base_preservation ^
  --lambda-preserve 1.0 --temperature 1.0 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 ^
  --seed 42 --train-ratio 0.8 --smoke-test

python -m src.training.train_gsi_paddy ^
  --org-dir "%ORG%" --prepared-dir "%PREPARED%" --base-model "%BASE%" ^
  --output-dir "%RESULTS%\pilot" --training-mode base_preservation ^
  --lambda-preserve 1.0 --temperature 1.0 --epochs 1 ^
  --device cpu --batch-size 1 --num-threads 2 --learning-rate 1e-4 ^
  --seed 42 --train-ratio 0.8
```

preflightも全ファイルのaudit/hashと同じsplit、両checkpointのstrict loadを実行します。
各splitの最初の1 batchをeval/no_gradでforwardし、combined objectiveを確認。
optimizer更新・checkpoint保存なし、status=preflight_passed、`preflight_metrics`に結果を記録。
backward時のRAMや全画像のmodel forward成功までは保証しません。

smokeも全auditと通常splitを維持し、最初のtraining batchで**optimizer.stepを1回のみ**、
validation 1 batchを処理します。epochs=1を要求し、status=smoke_test_completed、
run_purpose=smoke_test、max_batches_per_split=1としてsubset runを明示します。
epoch/best/finalも保存しますが、Pilot評価用checkpointではありません。
3回とも元Baseから別runを作り、preflight/smoke出力からPilotをresumeしません。

## 検証と限界

```bat
python -m pytest -q
python -m compileall -q run_poc.py src tests
```

testsは合成logits / 合成PNG / TinyModelを使用し、実重みは不要です。
loss mask・独立mean・KL方向/T²・λ=0・teacher固定/BN/no_grad/optimizer除外・
同一tensor・padding除外・epoch集計・provenance・preflight/smoke・strict loader roundtrip・
既存v0.1/SACLAJ regressionを確認します。実Base / 実GSI training成功の証明ではありません。

- Base teacherの誤りも保存する可能性があります。
- GSI unknownには未ラベルのAgricultureがあり得て、その画素ではpreservationがadaptationを妨げ得ます。
- SACLAJはpoint referenceで、pixel-perfect GTではありません。
- GSI latest imageryとSACLAJ observation dateのtemporal mismatchがあり得ます。
- λ=1.0 / T=1.0はPilot条件であり、最適値ではありません。
- v0.2はproduction modelではありません。CPU時間・RAMは会社PCで確認が必要です。
  teacherとstudentの2モデルを保持し、teacher forwardも追加されます。
- 従来同様、image-level splitはspatial independenceを保証しません。

[SACLAJ Evaluation v0.1](SACLAJ_EVALUATION.md)で元Baseと**Pilot**のbest.pthを比較し、
riceだけでなくAgriculture leakage、森林各subtype、Waterを確認してください。
v0.1と同じmapping/seed/抽出条件を用い、別runでGSI写真が更新され得る点も考慮します。
SACLAJ実データをtrainingに使わず、CSV/座標/地点別結果をGitや外部へ追加しません。

confidence threshold、selective preservation、spatial buffer、pseudo labels、DCHM、
tree species polygons、MAFF parcels、GSI conifer、manual GT、replay、λ/T探索、
3 epoch Pilot、GUI/exe、production claimsは今回追加しません。
