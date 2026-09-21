# Current Status

Updated: 2026-09-21

## Project goal

GSI航空写真から土地被覆を推論し、QGISで編集可能なGIS初期案を生成する。
成功の基準は学術的な精度だけではなく、AI出力によって実務上の人手による判読・図化の
負担を削減できるかどうかである。

## Phase A: Closed

Phase Aは **Closed** であり、milestone reviewの判定は **Conditional Pass** である。
manual GTを作らず、GSI teacher preparation、fine-tuning、SACLAJ development evaluation、
preservation手法の比較まで行うPhase Aは完了した。現在のlearning-method candidateは
v0.3 preservation-only replayだが、production modelではない。

Phase Aでは次を確認した。

- positive-only学習は、教師対象外のクラスを崩壊させる可能性がある。
- Base-preservationは、その崩壊を緩和した。
- preservation-only replayは、固定SACLAJ development sample上で適応と保持のtrade-offを改善した。

詳細は[Phase A Minimal End-to-End実験記録](PHASE_A_MINIMAL_E2E_SUMMARY.md)を参照すること。

## Phase B: multi-teacher pilots completed

Phase Bでは次の段階まで実データPilotと評価を実施済みである。

1. Paddy-only
2. Paddy + Water
3. Paddy + Water + Road

Water追加ではWater agreementの改善と既存class保持のtrade-offを確認した。Road teacher追加では
Road agreementが改善した一方、広範なnon-local class transition / interferenceも確認した。
これは局所的なRoad対Pavementの問題だけでは説明できず、teacherを逐次追加する現在の学習設定に
class間干渉があることを示す診断結果である。

そのため、現時点ではBuilding / Tree等のteacher追加や広範なbeta tuningへ進まない。
次の優先事項はobjective、replay構成、source間masking、gradient conflict等の診断である。
Phase Bの実行条件は[GSI Phase B training](GSI_PHASE_B_TRAINING.md)、隣接stage間の診断方法は
[classification transition diagnostics](CLASSIFICATION_TRANSITION_DIAGNOSTICS.md)を参照すること。

## New validation asset: year-matched GT54

OEM-SAR / DFC validation由来の日本54枚に含まれるmanual OEM8 GTについて、SARの
georeferenceを利用して地理参照済みGTを構築した。各地点のGSI年度別航空写真を人手で確認し、
対応年度を確定した。対象年度は **2007、2017、2018、2019、2020、2021** である。

地点、年度、regionの正式な再現性metadataは
[`manifests/val_gt_georef.csv`](../manifests/val_gt_georef.csv)として管理する。
`src/build_gsi_val_gt54.py` は指定年度のGSI RGBをGT gridへreprojectし、year-matched validation
datasetを構築する。会社PCの実データrunでは次を確認済みである。

- **54 / 54 items PASS**
- generated `manifest.csv` の **`alignment_ok=True` を全54件**
- real GSI tile downloadと、year-matched GSI RGB + georeferenced OEM8 GTの生成成功

画像、GT、生成dataset、QC出力はGit管理外である。このGT54を今後の主要evaluation assetとする。
手順と制約は[GSI年度別航空写真 + OEM-SAR validation GT 54枚](GSI_OEMSAR_VAL_GT54.md)を参照すること。

## Evaluation status

SACLAJ 1,000 pointsはdevelopment evaluationであり、final holdoutではない。GT54は今後の主要な
独立evaluation assetとして使用するが、公開・配布可否および評価protocolは別途確定する。
一般的なaccuracy improvement、全国generalization、production readinessは現時点では主張しない。

## Current constraints / unresolved issues

- Phase Bで観測したnon-local class interferenceの原因は未確定である。
- objective / replay / gradient conflictの診断と、独立した評価protocolの確定が必要である。
- SACLAJはfinal holdoutではない。
- Base modelのlicensingおよびenterprise-use条件は未解決である。
- GSI tileおよびOEM-SAR GTを組み合わせたdatasetの公開・再配布条件は未整理である。
- ローカルのrestricted data、モデル重み、checkpoint、生成outputはGitHubへ追加しない。
- docsのカテゴリ別再編（`status/`、`datasets/`、`training/`、`evaluation/`、`archive/`）は、
  リンク切れを避けるため今回行わず、将来のhousekeeping候補として残す。
