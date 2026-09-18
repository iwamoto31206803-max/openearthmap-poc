# GSI Phase B v0.1: Water-positive supervision

## 目的と状態

Phase A v0.3 preservation-only replayを維持したまま、GSI Water positiveをOEM8 class 6へ
明示的に教師化する最小の実験経路である。実装とsynthetic testのみ完了しており、実データの
preflight、training、SACLAJ evaluationは未実行である。結果や精度改善を示す文書ではない。

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
Waterあり: L_paddy + L_water + alpha * L_replay
Waterなし: L_paddy + alpha * L_replay
```

## company PCでの実行例

Python 3.11の既存環境で、まずSHA256を指定したpreflightを行う。以下のパスは公開用の例であり、
実際のrestricted data path、ID、出力をrepositoryへcommitしないこと。

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --preflight
```

preflightを監査した後の1 epoch Pilot:

```cmd
python -m src.training.train_gsi_phase_b ^
  --org-dir <PADDY_ORG> --prepared-dir <PADDY_PREPARED> ^
  --water-org-dir <WATER_ORG> --water-prepared-dir <WATER_PREPARED> ^
  --base-model <ORIGINAL_BASE_PTH> --base-sha256 <EXPECTED_SHA256> ^
  --epochs 1 --device cuda
```

`--water-*`が必須の独立entrypointなので、Phase Aの既存CLIでWaterが暗黙に有効になることはない。
Phase Bはbatch size 1、seed 42、ratio 0.8を検査する。

## Validationとmanifest

validationはPaddy positive、Water positive、Paddy replayを各全件1回評価する。Paddy/Waterごとに
positive CE、対象class agreement、対象class mean probability、unknown preservation KL、unknownの
student/Base argmax agreementを記録する。replayはpreservation KLとstudent/Base argmax agreementを
記録する。best checkpointはこれらsource objectiveの合計loss最小（同値は先のepoch）で選ぶ。

manifestにはexperiment/mode、Original Base path/SHA256、student/teacher初期化SHA、git SHA、seed、
lambda/T/alpha、optimizer、epoch/update数、全source件数、Water all-ignore未使用、class IDs、schedule、
split IDファイルとそのSHA256、prepared manifest SHA256、best/final checkpoint SHA256を記録する。
IDファイルに座標や画像内容は格納しない。run directory自体も`.gitignore`で保護される。

## 制約

実データ件数、入力監査、GPU memory、所要時間、Base hash、Water scheduleの実run inventory、
checkpoint再読込、SACLAJ/QGIS上の変化は **company-PC Pilotで要確認** である。model weights、GSI data、
GeoTIFF/GeoPackage、run outputs、SACLAJのCSV・座標・地点別結果・patchはGitへ追加しない。
