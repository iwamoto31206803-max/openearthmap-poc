# GT54 Evaluation Protocol v1.0

Status: Ready for implementation  
Project: OpenEarthMap PoC  
Target: A/B/C/D model comparison using GT54  
Scope: Evaluation protocol only

---

## 1. Purpose

本仕様は、GT54を用いてOpenEarthMap PoCのA/B/C/Dモデルを公平かつ再現可能に比較評価するためのルールを定義する。

本評価の目的は、単一のoverall scoreによるモデル順位付けではない。

特に以下を定量化する。

1. 各モデルのOEM8 class別性能
2. Paddy / Water / Road teacher追加による改善
3. teacher追加に伴う他classへのregression
4. Phase Bで観測されたnon-local class transitionの実態
5. tile / region / globalでの性能差
6. GT54 spatial overlapを考慮した重複排除評価

本仕様ではモデルのpass/fail thresholdやproduction採用基準は定義しない。

---

## 2. Evaluation Asset

評価にはGT54を使用する。

### 2.1 Dataset composition

GT54は以下の8地域、54 ValAreaから構成される。

| Region | Year | Tiles |
|---|---:|---:|
| Tokyo_Shijuku | 2019 | 12 |
| Tokyo_Haneda | 2019 | 6 |
| Fukushima_Daiichi | 2018 | 6 |
| Kanazawa | 2007 | 6 |
| Toyohashi | 2020 | 6 |
| Moriguchi_Yodogawa | 2021 | 6 |
| Osaka_Port | 2017 | 6 |
| Mozu | 2021 | 6 |

Total: 54 tiles / 8 regions

### 2.2 RGB

評価RGBはOEM-SAR validation optical RGBそのものではない。

各ValAreaについて、対応年度のGSI航空写真から再構築したRGBを使用する。

処理:

GSI year-specific XYZ tiles  
→ mosaic  
→ GT gridへのbilinear reprojection

GT gridを基準とし、GT自体はresampleしない。

RGB / GTは以下が一致していることを前提とする。

- CRS
- transform
- width
- height
- bounds

Full run QC:

- 54 / 54 PASS
- alignment_ok=True for all items

### 2.3 Manual GT

GTはmanual OEM8 class labelsであり、class ID 0–8を保持する。

GT class IDは再マッピングしない。

### 2.4 Provenance limitation

Original Base checkpointのtraining provenanceは完全には確認されていない。

特に以下は未確認である。

- GT54 sceneとのtraining overlap
- source-image overlap
- OEM / OEM-SAR training dataとの関係

したがってGT54を以下の名称では扱わない。

- independent holdout
- independent test set
- independent evaluation set
- unseen test set

使用可能な表現例:

- full-scene manual-GT evaluation asset
- quantitative evaluation asset
- year-matched validation asset
- GT54 quantitative evaluation on year-matched GSI imagery

---

## 3. OEM8 Classes

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Bareland |
| 2 | Grass/Rangeland |
| 3 | Pavement/Developed space |
| 4 | Road |
| 5 | Tree |
| 6 | Water |
| 7 | Agriculture/Cropland |
| 8 | Buildings |

重点評価class:

- 3 Pavement / Developed space
- 4 Road
- 5 Tree
- 6 Water
- 7 Agriculture / Cropland
- 8 Buildings

---

## 4. Evaluation Models

### A. Original Base

SHA256:

852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c

### B. Paddy-only

SHA256:

e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172

### C. Paddy + Water

beta_water = 0.5

SHA256:

ff721d91959da847adee2d26639384f03bfa4ded5ea118e40ae16576260c4b6e

### D. Paddy + Water + Road

beta_water = 0.5  
beta_road = 1.0

SHA256:

b4abe64d86e5b4350be2107d154a547f6b87b48e884ea87eac9e088cec201ed9

### 4.1 Interpretation

A/B/C/Dはliteralな逐次continuation modelではない。

B/C/DはそれぞれOriginal Baseを開始点とするteacher-composition variantsとして扱う。

---

## 5. Fair Comparison Conditions

A/B/C/Dは完全に同一の評価条件で比較する。

以下は全モデルで固定する。

- RGB input
- GT
- preprocessing
- inference image handling
- resize / padding
- tile treatment
- output resolution
- argmax rule
- nodata handling
- evaluation mask
- spatial de-duplication ownership

checkpoint以外の評価条件をモデルごとに変更してはならない。

---

## 6. Background Class 0

Class 0 BackgroundはPrimary semantic metricから除外する。

Primary semantic evaluation対象:

Classes 1–8

したがって以下を原則とする。

- per-class IoU: classes 1–8
- per-class Precision: classes 1–8
- per-class Recall: classes 1–8
- mIoU-8: classes 1–8
- Pixel Accuracy: GT classes 1–8

ただしclass 0は診断目的で保持する。

少なくとも以下を記録する。

- GT class 0 support
- prediction class 0 support
- semantic class → 0
- 0 → semantic class

Confusion matrixは0–8の9 classを保持する。

Class 0 Backgroundとraster NoDataは別物として扱う。

- Class 0: 有効なGT label
- NoData: evaluation対象外

---

## 7. Per-Class Metrics

各classについて以下を計算する。

TP_c  
FP_c  
FN_c

### IoU

IoU_c = TP_c / (TP_c + FP_c + FN_c)

### Precision

Precision_c = TP_c / (TP_c + FP_c)

### Recall

Recall_c = TP_c / (TP_c + FN_c)

### Support

GT_support_c = TP_c + FN_c

Prediction_support_c = TP_c + FP_c

---

## 8. Absent Class Rule

評価単位内でGT support = 0のclassについては、そのclassのIoU / Recallを0としない。

原則:

GT support = 0:

- IoU = NA
- Recall = NA
- GT support = 0
- Prediction supportを記録
- False-positive pixelsを記録

GT support > 0 かつ prediction support = 0:

- IoU = 0
- Recall = 0
- Precision = NA

Macro averageではGT support > 0のclassのみを対象とする。

Minimum support thresholdは設けない。

低support classは除外せず、support値を必ず併記する。

これは意図的なreporting conventionであり、
GT support = 0 かつ FP > 0の場合にIoU=0を機械的に割り当てる一般的実装とは異なる。

Absent classへのfalse-positive burdenは、IoUではなくPrediction supportおよびFP countで確認する。

---

## 9. Aggregate Metrics

### 9.1 Pixel Accuracy

GT class 1–8のvalid pixelsについて算出する。

Pixel Accuracyはclass imbalanceの影響を強く受けるため、Primary single scoreとはしない。

### 9.2 mIoU-8

Classes 1–8について、当該評価単位でGT support > 0のclassのIoUをmacro averageする。

### 9.3 Macro Precision / Recall

Macro PrecisionおよびMacro RecallはPrimary overall metricとして定義しない。

Precision / Recallは以下の粒度で使用する。

- class-level
- region-level
- comparison delta

### 9.4 Interpretation

mIoU上昇のみをもってmodel improvementとは判定しない。

重点classの個別性能とregressionを必ず併記する。

---

## 10. Evaluation Hierarchy

評価は以下の5層で行う。

### 10.1 Tile-level

54 ValAreaそれぞれを個別評価する。

用途:

- scene-specific failure detection
- outlier detection
- local transition analysis

位置づけ:

Diagnostic

Tile-level mIoUの単純平均を「GT54 mIoU」と呼ばない。

---

### 10.2 Region-level de-duplicated

8地域それぞれについて、region内の重複pixelを除外して評価する。

各regionでは以下を算出する。

- per-class IoU
- Precision
- Recall
- GT support
- Prediction support
- mIoU-8
- Pixel Accuracy
- confusion matrix

位置づけ:

Primary building block

---

### 10.3 Global raw

54 tileをそのまま集計する。

Spatial overlapによる同一地理pixelの重複計数を許容する。

位置づけ:

Secondary / reference

用途:

- conventional pooled metric
- comparison with prior pipelines
- quantification of overlap effect

Global rawをPrimary metricとして使用しない。

---

### 10.4 Global de-duplicated

GT54全体について、同一地理pixelを1回のみ評価する。

位置づけ:

Primary

主な出力:

- mIoU-8
- Pixel Accuracy
- classes 1–8 IoU
- classes 1–8 Precision / Recall
- confusion matrix
- support

---

### 10.5 Equal-region macro

Primary region-balanced metricは、
Class-first Equal-region Macro mIoUとする。

各class cについて:

RegionMacroIoU_c =
mean(IoU_c,r)

ただし、そのclassのGT supportが存在するregionのみ対象とする。

その後、

RegionMacro_mIoU =
mean(RegionMacroIoU_1, ..., RegionMacroIoU_8)

とする。

Shijukuのtile数が他regionの2倍であることによる重み偏りを回避する。

以下は別の診断値として扱う。

Mean Regional mIoU =
mean(region-level mIoU)

Mean Regional mIoUはPrimary metricではない。

---

## 11. Primary and Secondary Results

### Primary

1. Global de-duplicated
2. Class-first Equal-region Macro

### Secondary

3. Global raw

### Diagnostic

4. Region-level
5. Tile-level
6. Mean Regional mIoU
7. Confusion matrix
8. Transition analysis
9. Correction / regression analysis

---

## 12. Spatial De-duplication Preflight

Pixel-level de-duplicationの前に、重複tile間のgrid compatibilityを検証する。

少なくとも以下を確認する。

- CRS equality
- pixel size equality
- no raster rotation / shear
- grid phase / origin compatibility
- lossless mapping to a common pixel lattice

### 12.1 Preflight PASS

すべてのoverlapping GT tilesが共通pixel latticeへlosslessに対応可能な場合、
pixel-level de-duplicationへ進む。

### 12.2 Preflight FAIL

重複tile間でlosslessな共通pixel latticeを定義できない場合、

- GTをresampleしない
- nearest-neighbor等で強制整合しない
- Global de-duplicated評価を実行しない
- protocol / data handlingを再検討する

GT gridを評価の正とする原則を優先する。

---

## 13. Canonical Pixel Identity

Spatial preflight PASS後、GT54全体にcanonical integer grid indexを定義する。

同一地理pixelのidentityはfloating-point coordinatesではなく、
canonical integer indexで表現する。

概念的には:

global_col = integer index from canonical origin and pixel width  
global_row = integer index from canonical origin and pixel height

Pixel key:

(global_row, global_col)

Floating-point座標文字列の直接比較は使用しない。

Canonical latticeの定義とrounding / tolerance ruleは実装時に明示し、
preflightでlossless compatibilityを確認する。

---

## 14. GT Conflict Rule

同一canonical pixelに複数GT observationが存在する場合、
valid GT class ID集合を比較する。

### Consistent GT

すべて同一classの場合:

- consistent overlap
- GT labelを採用可能

### GT_CONFLICT

2種類以上のvalid class IDが存在する場合:

GT_CONFLICT

とする。

Class 0 Backgroundもvalid classとしてconflict判定に含める。

GT_CONFLICT pixelはGlobal / Region de-duplicated metricから除外する。

以下をQCとして記録する。

- n_overlap_pixels
- n_consistent_overlap_pixels
- n_gt_conflict_pixels
- gt_conflict_rate
- conflict class combinations
- involved regions
- involved tiles

GT conflictが想定以上に多い場合は、
モデル評価より先にevaluation asset品質を再確認する。

---

## 15. Prediction Ownership

GT-consistent overlap pixelについて、prediction ownerを一意に決定する。

Owner determinationは以下の情報のみに依存する。

- tile geometry
- canonical grid
- pixel position relative to tile edge
- deterministic tie-break rule

以下はowner selectionに使用しない。

- model prediction
- GT class
- checkpoint identity

### 15.1 Interior distance

各pixelについて:

interior_distance =
min(
distance_to_left,
distance_to_right,
distance_to_top,
distance_to_bottom
)

Interior distanceが最大のtileをownerとする。

### 15.2 Tie

Interior distanceが同一の場合は、
固定されたValArea ID orderingで決定する。

### 15.3 Common owner map

同一owner mapをA/B/C/Dすべてに適用する。

Majority voteは使用しない。

理由:

- artificial ensemble effectを避ける
- deterministic evaluationを維持する
- tile-edge artifactの影響を抑える
- 全モデルを完全に同一pixel集合で比較する

---

## 16. Formal Model Comparisons

正式比較は以下の5組とする。

### Base-relative comparisons

- A → B
- A → C
- A → D

目的:

Original Baseからのtotal effectを評価する。

### Incremental teacher comparisons

- B → C
- C → D

目的:

特定teacher追加によるincremental effectを評価する。

特にC → DはRoad teacher追加の効果と副作用を評価する主要比較とする。

---

## 17. Formal Comparison Surface

正式な以下の解析はGlobal de-duplicated pixel set上で行う。

- correction / regression analysis
- prediction transition analysis
- GT-conditioned transition analysis
- net correct change

これによりspatial overlapによる二重計数を避ける。

Region-level de-duplicated transitionは診断目的で算出可能とする。

Tile-level / raw transitionは補助出力として扱い、
正式なglobal comparison resultとはしない。

---

## 18. Delta Metrics

各formal comparisonについてclasses 1–8で以下を算出する。

- ΔIoU
- ΔPrecision
- ΔRecall

重点6 classでは追加で以下を算出する。

- Corrected pixels
- Regressed pixels
- Changed-but-still-wrong pixels
- Net correct change
- Corrected rate
- Regressed rate
- Net correct rate

Net correct change:

Corrected - Regressed

Net correct rate:

(Corrected - Regressed) / GT support

---

## 19. Pixel-Level Change Categories

Before modelとAfter modelを比較し、各evaluation pixelを以下に分類する。

### Unchanged Correct

before = GT  
after = GT

### Correction

before != GT  
after = GT

### Regression

before = GT  
after != GT

### Changed-but-still-wrong

before != GT  
after != GT  
before != after

### Unchanged Wrong

before != GT  
after != GT  
before = after

---

## 20. Prediction Transition Analysis

各formal comparisonについて、

before prediction → after prediction

のtransition matrixを算出する。

さらにGTを加え、

GT / before prediction / after prediction

の3-way集計を行う。

これによりtransitionを、

- correction
- regression
- changed-but-still-wrong

へ分解する。

---

## 21. Fixed Transition Diagnostics

特に以下のtransitionを固定診断対象とする。

- Building → Road
- Tree → Road
- Water → Road
- Agriculture → Road
- Pavement → Road
- Road → Pavement

加えて各comparisonについてprediction transition数上位N件を抽出する。

Initial default:

N = 10

---

## 22. Benefit / Regression Reporting

単一composite scoreは作成しない。

特定teacher追加について、

BenefitとRegressionを分離して報告する。

例: C → D

### Road benefit

- Road ΔIoU
- Road ΔPrecision
- Road ΔRecall
- Road corrected pixels
- Road corrected rate
- Road net correct rate

### Potential regressions

- Pavement ΔIoU
- Tree ΔIoU
- Water ΔIoU
- Agriculture ΔIoU
- Buildings ΔIoU
- regressed pixels / rates

Overall mIoUのみを根拠にmodel improvementとは記述しない。

---

## 23. Statistical Uncertainty

GT54 tilesは統計的に独立な54 samplesとは扱わない。

理由:

- within-region spatial overlap
- common geographic context
- unequal tile composition

Tile-level t-test等をPrimary inferenceに使用しない。

### 23.1 Required region summaries

Formal comparisonのregion-level deltaについて、
必要に応じて以下を報告する。

- median
- IQR
- min
- max

### 23.2 Bootstrap

Region-level bootstrapはv1.0の必須要件としない。

必要に応じて、
8 regionをsampling unitとするoptional statistical extensionとして実施できる。

Bootstrap CIはGT54内でのregion variabilityの補助的指標とする。

GT54外へのgeneralization guaranteeとして解釈しない。

---

## 24. Required QC

Evaluation runごとに少なくとも以下を確認する。

### Dataset QC

- expected item count = 54
- expected region count = 8
- RGB / GT alignment
- CRS equality
- transform equality
- width / height equality
- bounds equality
- valid class IDs = 0–8
- NoData handling

### Cross-tile grid QC

- overlapping tile CRS compatibility
- pixel size compatibility
- rotation / shear
- grid phase compatibility
- canonical lattice compatibility
- preflight PASS / FAIL

### Class QC

- GT class histogram
- prediction class histogram
- class 0 support
- absent classes
- low-support classes

### Spatial QC

- overlap pixel count
- consistent overlap count
- GT conflict count
- GT conflict rate
- owner map determinism
- evaluated de-duplicated pixel count

### Model QC

- checkpoint SHA256
- model architecture
- preprocessing identity
- inference configuration identity

---

## 25. Required Outputs

Evaluatorは少なくとも以下を出力する。

### Summary

- evaluation_summary.csv
- model_summary.csv

### Class metrics

- global_deduplicated_class_metrics.csv
- global_raw_class_metrics.csv
- region_class_metrics.csv
- tile_class_metrics.csv

### Confusion

- confusion_global_deduplicated.csv
- confusion_global_raw.csv
- confusion_by_region.csv

### Comparisons

- model_comparison_metrics.csv
- correction_regression.csv
- prediction_transitions.csv
- gt_conditioned_transitions.csv

### QC

- dataset_qc.csv
- grid_preflight_qc.csv
- overlap_qc.csv
- gt_conflict_pixels.csv
- owner_map_summary.csv

Exact filenames may be adjusted at implementation review,
but output semantics must remain fixed。

---

## 26. Reporting Structure

Formal evaluation reportは以下の順序を推奨する。

1. Evaluation conditions
2. Dataset / model identity
3. Grid preflight result
4. Primary Global de-duplicated results
5. Primary Equal-region macro results
6. Region-level results
7. A/B/C/D model comparison
8. Correction / regression analysis
9. Transition analysis
10. Tile-level diagnostics
11. Raw vs de-duplicated comparison
12. QC
13. Limitations / caveats

---

## 27. Mandatory Limitations

Result interpretationには以下を明示する。

### Reconstructed RGB

評価RGBはOEM-SAR official validation optical imageryではなく、
year-matched GSI aerial imagery reconstructionである。

### Training provenance

Original Base training dataとGT54とのindependenceは確認されていない。

### Spatial composition

54 tilesは54個の完全独立sceneではなく、
within-region spatial overlapを含む。

### Geographic representativeness

GT54は日本全国を統計的に代表するsampling designではない。

したがってGT54 performanceを日本全国でのproduction performanceと同一視しない。

---

## 28. Explicit Non-Goals in v1.0

v1.0では以下を採用しない。

- model ranking
- single composite score
- production pass/fail threshold
- arbitrary minimum class support threshold
- tile-level significance testing
- majority-vote de-duplication
- prediction-dependent ownership
- GT-dependent ownership
- forced GT resampling for de-duplication
- post-hoc class selection
- independent test setという表現

---

## 29. Interpretation Principle

GT54 evaluationの中心は、

「overall accuracyが上がったか」

だけではない。

以下を同時に評価する。

1. Target classが改善したか
2. 改善が複数regionで再現するか
3. 他classの性能を損なっていないか
4. teacher追加によるprediction transitionがGTに照らしてcorrectionかregressionか
5. spatial overlapを除いても同じ傾向が維持されるか

特にRoad teacher追加については、

C → D

を用いてRoad benefitと、

- Building
- Tree
- Water
- Agriculture
- Pavement

へのregressionを同時に評価する。

---

## 30. Implementation Gate

Evaluator実装へ進む前に、本仕様について少なくとも以下を確認する。

- Background class rule
- NoData rule
- absent class rule
- metric definitions
- class-first Equal-region macro definition
- cross-tile grid preflight
- canonical pixel identity
- de-duplication ownership
- GT conflict handling
- formal comparison pairs
- formal comparison surface
- correction / regression definitions
- required outputs
- caveat wording

上記が合意された後、Codexへevaluator実装を依頼する。
