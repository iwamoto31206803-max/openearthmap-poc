# GSI Phase B v0.1–v0.3: weighted Water and Road-positive supervision

## 目的と状態

Phase A v0.3 preservation-only replayを維持したまま、GSI Water positiveをOEM8 class 6へ
明示的に教師化する実験経路である。Phase B v0.1のSACLAJ development evaluationではWaterが
89%まで改善した一方、Agricultureと一部Tree subtypeにtrade-offが見られた。v0.2はそのbalanceを
確認するため、**Water source objective全体のweightだけを0.5へ下げた**。v0.1 / v0.2の
実データPilotとSACLAJ development evaluationは完了している。SACLAJはdevelopment evaluationであり、
manual GTまたはproduction accuracyの根拠ではない。

Phase B v0.3はv0.2をそのまま対照条件とし、**Road positive teacherだけ**を追加する未実行の
Pilotである。`beta_water=0.5`を維持し、`beta_road=1.0`とする。この値が最適とは仮定しない。

## v0.3 Road Pilot仕様

Road auditは2,000 images、positive-bearing 1,639、all-ignore 361、positive pixels
65,289,637（9.977510666780771%）、pairing valid、size mismatch 0である。exact-red以外の
261 mismatch pixelsはteacher maskに含まれない。Road labelはOEM8 class 4またはignore 255だけを
受理し、all-ignore 361件はtraining、replay、validationのいずれにも使用しない。

Road positive pixels上のOriginal Base argmaxはRoad 55.5998%、Pavement / Developed space
33.0333%であり、mean probabilityはそれぞれ0.525316、0.334826である。従って主なBase conflictは
**Road vs Pavement / Developed space**である。これは診断値であり、Road精度や境界問題の解決を
示すものではない。

PaddyとRoadにはsource image SHA256でbyte-identicalな7 images（Paddy positive train 6、Paddy
replay train 1）がある。Water/Road overlapは0である。v0.3ではfilenameやnumeric IDではなく画像
content SHA256で、この7件をRoad positive candidate poolから**split前に除外**する。Paddy splitと
replay splitは変えない。manifestには件数と、raw hashを露出しない決定的なsanitized referenceを
記録する。referenceは`sha256("road-overlap:" + source SHA256)`の先頭16桁を使うため、filenameや
numeric IDに依存せず、同じsource imageをrun間で再照合できる。source間positive/unknown競合を
一般化して解決するmulti-teacher-aware maskingは今回
導入せず、future workとする。

除外後のRoad candidateをseed 42、既存80/20規則でsplitし、Road train poolからseed 42で重複なく
ちょうど1,028 samplesを選ぶ。各Paddy logical stepでRoadを1件ずつ消費するため、Roadは追加の
optimizer updateを作らない。未使用train数をmanifestへ記録し、validationはRoad validation全件を
各1回評価する。

```text
L_road = Road positive CE(class 4) + lambda * T^2 * Road unknown KL(Base || Student)
Waterあり: L_paddy + beta_water * L_water + beta_road * L_road + alpha * L_replay
Waterなし: L_paddy + beta_road * L_road + alpha * L_replay
```

Road source metricsはraw positive CE、class-4 agreement/mean probability、unknown preservation
KL、unknown student/Base argmax agreement、positive/unknown pixel countsである。`beta_road`は
validation totalを含むtotal objectiveだけへ掛け、raw metricsには掛けない。progress logにも
Road consumed/totalとrunning raw Road CEを加える。

## 固定仕様

- studentとfrozen/eval teacherは、ともに同じ **Original Base** から別々に構築する。Phase A
  checkpointからresumeしない。
- Paddy positiveはclass 7 CE、Paddy unknownは`KL(Base || Student)`、Paddy all-ignoreは
  preservation-only replayとし、Phase A v0.3のsplitとexposureを維持する。
- Water positiveはclass 6 CE、Water unknownは`KL(Base || Student)`とする。unknownは
  Water-negativeではない。
- Water all-ignore 558件は件数を監査・manifestへ記録するだけで、training、replay、validationの
  いずれにも使わない。最初のPilotで正解として解釈できず、v0.3のPaddy replayとの比較を
  不必要に変えないためである。
- seed 42、ratio 0.8の既存の切り捨て規則により、Water positiveは想定692件から
  train 553 / validation 139となる。

1 epochはPaddy positive trainの1,028 logical stepsである。Water trainの各sampleは、seed 42で
`random.sample`した553 slotsへ決定的に分散し、split順に各1回だけ消費する。cycle/repeatせず、
残る475 stepsはWaterなしである。scheduleはWaterを挿入するslot集合だけを保持し、各selected slotで
Water loaderから次のsampleを消費する。Paddy replayはv0.3と同じscheduling semanticsを維持し、
loaderがepoch axisより短い場合のみ決定的に再開する。現行Pilotでは1051 > 1028のため再開しない。
この設計によりWater追加後もoptimizer updatesとPaddy/replay exposureを変えない。

各stepのlossは次の単純加算で、`backward()`と`optimizer.step()`は各1回だけである。Paddyと
Waterを平均しない（既定`lambda=1, T=1, alpha=1`）。

```text
L_source = mean positive CE + lambda * T^2 * mean unknown KL(Base || Student)
Waterあり: L_paddy + beta_water * L_water + alpha * L_replay
Waterなし: L_paddy + alpha * L_replay
```

`beta_water`はWater positive CEとWater unknown preservation KLを含む`L_water`全体へ掛かる。
raw Water CE/KL metricsには掛けない。CLI既定値`1.0`はv0.1を再現し、v0.2 Pilotでは`0.5`を
明示する。Paddy、replay、split、schedule、1,028 updatesその他の条件はv0.1から変更しない。

## company PCでの実行例

Python 3.11の既存環境で、まずSHA256を指定したpreflightを行う。以下のパスは公開用の例であり、
実際のrestricted data path、ID、出力をrepositoryへcommitしないこと。

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --preflight
```

短いsmoke test:

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --smoke-test
```

preflightとsmoke testを監査した後の1 epoch Pilot:

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --epochs 1 --device cuda
```

`--water-*`が必須の独立entrypointなので、Phase Aの既存CLIでWaterが暗黙に有効になることはない。
Phase Bはbatch size 1、seed 42、ratio 0.8を検査する。

Road v0.3 Pilotでは次の3段階で明示的にRoad sourceを有効にする（実データPilotはまだ未実行）。

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --road-org-dir <ROAD_ORG> --road-prepared-dir <ROAD_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --beta-road 1.0 --preflight

python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --road-org-dir <ROAD_ORG> --road-prepared-dir <ROAD_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --beta-road 1.0 --smoke-test

python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --road-org-dir <ROAD_ORG> --road-prepared-dir <ROAD_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --beta-water 0.5 --beta-road 1.0 --epochs 1 --device cuda
```

Road引数を省略すればv0.1/v0.2経路を維持する。Road引数は必ずorg/preparedの組で指定し、
`beta_road`はfiniteかつ0以上でなければならない。

## Validationとmanifest

validationはPaddy positive、Water positive、Paddy replayを各全件1回評価する。Paddy/Waterごとに
positive CE、対象class agreement、対象class mean probability、unknown preservation KL、unknownの
student/Base argmax agreementを記録する。replayはpreservation KLとstudent/Base argmax agreementを
記録する。best checkpointはこれらsource objectiveの合計loss最小（同値は先のepoch）で選ぶ。

manifestにはexperiment/mode、Original Base path/SHA256、student/teacher初期化SHA、git SHA、seed、
lambda/T/alpha、`beta_water`、loss expression、model version、optimizer、epoch/update数、全source件数、Water all-ignore未使用、class IDs、schedule、
split IDファイルとそのSHA256、prepared manifest SHA256、best/final checkpoint SHA256を記録する。
IDファイルに座標や画像内容は格納しない。run directory自体も`.gitignore`で保護される。

training中は50 logical stepsごとにstep/total、消費したWater数、running Paddy CE、raw Water CE、
replay KLを表示する。validationではこのprogress logを出さず、IDやpathも表示しない。

## 実データPilot結果

v0.1は`beta_water=1.0`、v0.2は`beta_water=0.5`である。両者ともOriginal Base start、
Paddy / Water positive supervision、Paddy all-ignore replay、Water all-ignore未使用とし、1,028
logical steps / optimizer updatesのうちWaterを553 stepsで使用した。v0.2ではWater positive CEと
Water unknown Base-preservation KLを含むWater source objective全体に0.5を掛けた。

v0.2 training run `20260919T053943_488504Z_000833f3`は`completed`で、experimentは
`gsi_phase_b_v0.2`、best checkpoint SHA256は
`ff721d91959da847adee2d26639384f03bfa4ded5ea118e40ae16576260c4b6e`である。validationは
Paddy agreement 0.848088094...、Paddy mean probability 0.578560966...、Water agreement
0.717916463...、Water mean probability 0.566900472...、replay preservation KL 0.476312183...であった。

SACLAJ development evaluationと入力identityの結果・解釈は
[Phase B Water milestone](PHASE_B_WATER_MILESTONE_20260919.md)に固定する。

## 制約

本結果はSACLAJ development evaluationであり、nationwide accuracy、generalization、production
readinessを示さない。v0.2の最適性やTree issueの解決も示さない。model weights、GSI data、
GeoTIFF/GeoPackage、run outputs、SACLAJのCSV・座標・地点別結果・patchはGitへ追加しない。
