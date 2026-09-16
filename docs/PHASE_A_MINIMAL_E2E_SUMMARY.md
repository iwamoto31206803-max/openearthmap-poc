# Phase A Minimal End-to-End 実験記録

## 1. 位置づけと目的

本記録は、manual ground truth（manual GT）を作成しないPhase Aにおいて、
`teacher preparation → fine-tuning → independent SACLAJ evaluation` という最小の
End-to-End経路が成立するかを確認した実験のまとめである。production modelの成果報告、
production accuracyの証明、またはモデル採用判断ではない。

目的は、GSI由来の水田partial labelをOEM8 Agricultureへ適応させ、その変更が既存の
OEM8出力へ及ぼす影響を、学習には使用していないSACLAJ地点referenceで比較可能な形に
することだった。OEM8の既存class IDは変更しておらず、対象はclass 7
（Cropland / Agriculture）である。

## 2. DatasetとBase model

GSI paddy partial-label datasetは、GSI航空写真2,600画像に対し、水田に該当する画素だけを
class 7のpositiveとして与え、その他をclass 255（unknown / ignore）としたデータである。
positiveは156,376,781 / 850,576,012画素（約18.38%）、全画素unknownのFALSE画像は
1,314枚だった。FALSE画像を除いた1,286枚を、seed 42でtrain 1,028枚 / validation
258枚に決定的に分割した。unknownはBackgroundでもnegativeでもない。また、この
image-level splitはspatial independenceを保証しない。

Base modelには `RGB_Real_5_u-efficientnet-b4.pth` を使用した。architectureは
9-classのSMP U-Net（EfficientNet-B4 encoder、scSE）で、入力はuint8 RGBをfloat32へ変換し
255で除算する既存前処理を維持した。encoderのparameterとtrain時のBN/dropoutを固定し、
decoderとsegmentation headのみを更新した。

## 3. v0.1 positive-only fine-tuning

v0.1はpositive画素だけに `CrossEntropyLoss(ignore_index=255)` を適用する設計である。
unknown画素にはlossを与えず、FALSE画像も除外した。これはGSI partial labelへの
adaptationを最小構成で確認するためのPilotであり、他classの保持を目的関数に含めていない。

同じSACLAJ標本で、Rice → AgricultureはBase 27%からv0.1 99%、Other crop →
Agricultureは24%から92%となった。一方、非Agriculture地点でAgricultureを出すleakageは
3.375%から39.25%へ変化し、Broadleaf / Needleleaf / Mixed → Treeはそれぞれ
89% / 83% / 87%から13% / 7% / 17%、Water → Waterは81%から6%となった。
この全体的な出力driftを、本実験では **Agriculture collapse** として扱った。
positive-onlyの高いpositive agreementだけでは、既存OEM8能力の保持を判断できないことが
明確になった。

## 4. Base-preservation仮説とv0.2

Agriculture collapseは、unknownをlossから完全に除外したことで、class 7以外のBase出力を
維持する制約がなかったためではないか、というBase-preservation仮説を立てた。v0.2では
v0.1 checkpointから継続せず、**original Baseから再スタート**し、次の目的関数を用いた。

- **positive:** GSI class-7 labelに対するcross entropy（GSI CE）
- **unknown:** `KL(Base teacher || student)`（9-class分布、teacherは固定）
- **合成:** `L_total = L_GSI + lambda × L_preserve`
- **Pilot条件:** `lambda=1.0`、`T=1.0`

positiveとunknownは別々の画素数で平均し、synthetic paddingは両方から除外した。teacherと
studentは同じoriginal Baseをstrict loadし、teacherはeval / no-gradで固定した。

会社PCでは、同じdataset、split、original Baseを用いたv0.2の **preflight、1-step
smoke、1 epoch Pilot** が順に完走した。preflightとsmokeはPilotとは別runで、いずれも
original Baseから開始し、resumeには使用していない。SACLAJ evaluationには1 epoch Pilotの
`checkpoints/best.pth`を用いた。

## 5. SACLAJ 1,000点での比較

評価は10 subtypeそれぞれ最大100点、合計1,000点の **stratified capped sample**
（seed 42）を同一条件で用いたBase / v0.1 / v0.2比較である。矢印の右側は各subtypeで期待する
OEM8 class、値はそのclassとのpoint agreementを示す。Agriculture leakageだけは、Tree / Water
の非Agriculture 800点でAgricultureを予測した割合であり、低い方向が望ましい。

| SACLAJ指標 | Base | v0.1 positive-only | v0.2 Base-preservation |
|---|---:|---:|---:|
| Rice → Agriculture | 27% | 99% | 84% |
| Other crop → Agriculture | 24% | 92% | 56% |
| Agriculture leakage | 3.375% | 39.25% | 6.5% |
| Broadleaf → Tree | 89% | 13% | 90% |
| Needleleaf → Tree | 83% | 7% | 89% |
| Mixed → Tree | 87% | 17% | 91% |
| Water → Water | 81% | 6% | 34% |
| Needleleaf evergreen → Tree | 88% | 14% | 96% |
| Broadleaf evergreen → Tree | 88% | 15% | 88% |
| Needleleaf deciduous → Tree | 85% | 6% | 96% |
| Broadleaf deciduous → Tree | 79% | 6% | 85% |

## 6. 解釈

- GSI partial labelsによるclass 7へのadaptation自体は成立した。
- positive-only supervisionは、既存OEM8能力を大きく破壊し得る。
- Base-preservationにより、v0.1で見られたAgriculture leakageとTree系degradationは大幅に
  改善した。
- Tree系はSACLAJ agreement上、ほぼBase水準を維持した。
- Waterはv0.1の6%から34%へ変化したが、Baseの81%には届かず、未解決である。
- v0.2はPhase Aの基本学習方式候補であるが、production modelではない。

これらは「accuracy improved」という主張ではない。SACLAJはfine-tuningに用いていない
independent point referenceだが、pixel-perfect GTではなく、training領域やBase pretrainingとの
地理的重複を検証した意味での統計的独立性も主張しない。各subtype最大100点の上限付き層化標本は
日本全体のclass頻度を表さない。SACLAJ観測時期とGSI latest imageryの撮影時期にはtemporal
mismatchがあり得て、実際の土地被覆変化も不一致に含まれ得る。また、softmax probabilityは
calibrated accuracyではない。

## 7. Run provenanceとlocal execution reference

実runのmanifest、checkpointおよび評価結果は会社PCローカルにあり、本リポジトリには含めて
いない。SACLAJのCSV、座標、地点ID、地点別結果も外部へ持ち出さない。このため、本記録作成時に
リポジトリ内で確認できた範囲では、実値のrun IDを転記・検証できない。未確認のIDを推測して
記載しない。

| 対象 | manifestで確認するfield / 主要checkpoint | 記録状況 |
|---|---|---|
| v0.1 1 epoch Pilot | `run_manifest.json` の `run_id`; `checkpoints/best.pth`, `checkpoints/final.pth` | company PC local execution reference（IDはrepository外） |
| v0.2 preflight | `run_manifest.json` の `run_id`, `status=preflight_passed` | company PC local execution reference（IDはrepository外） |
| v0.2 1-step smoke | `run_manifest.json` の `run_id`, `status=smoke_test_completed`; smoke用checkpoint | company PC local execution reference（IDはrepository外） |
| v0.2 1 epoch Pilot | `run_manifest.json` の `run_id`; `checkpoints/best.pth`, `checkpoints/final.pth` | company PC local execution reference（IDはrepository外） |
| SACLAJ Base / v0.1 evaluation | `evaluation_manifest.json` の `run_id` | company PC local execution reference（IDはrepository外） |
| SACLAJ Base / v0.2 evaluation | `evaluation_manifest.json` の `run_id` | company PC local execution reference（IDはrepository外） |

ローカル絶対pathは環境固有の **local execution reference** とし、再現性の識別には各manifestの
run ID、checkpoint SHA256、Base SHA256、prepared manifest SHA256、SACLAJ CSV / mapping SHA256を
使用する。将来この表を更新する場合も、SACLAJの機微情報やモデル重み・実行結果そのものはGitへ
追加せず、公開可能性を確認した識別情報だけを記録する。

実装と個別手順は、[Phase A v0.1](GSI_PHASE_A_TRAINING.md)、
[Base-Preservation v0.2](BASE_PRESERVATION_FINETUNING.md)、
[SACLAJ Evaluation v0.1](SACLAJ_EVALUATION.md)を参照すること。
