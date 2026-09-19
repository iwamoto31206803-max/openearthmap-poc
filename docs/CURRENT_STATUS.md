# Current Status

Updated: 2026-09-19

## Project goal

GSI航空写真から土地被覆を推論し、QGISで編集可能なGIS初期案を生成する。
成功の基準は学術的な精度だけではなく、AI出力によって実務上の人手による判読・図化の負担を削減できるかどうかである。

## Current phase

Phase Aは監査を含めて完了し、判定は **CONDITIONAL PASS** である。
現在のlearning-method candidateは **v0.3 preservation-only replay** だが、production modelではない。

Phase Aでは次を確認した。

- positive-only学習は、教師対象外のクラスを崩壊させる可能性がある。
- Base-preservationは、その崩壊を緩和した。
- preservation-only replayは、固定SACLAJ development sample上で適応と保持のtrade-offを改善した。

## Current direction

次の優先事項は、manual GTではなく **GSI-only expansion** である。

1. OEM8への意味的な対応が十分明確な公開GSI teacherを追加する。
2. positive weak label以外ではBaseの挙動を保持する。
3. SACLAJはdevelopment evaluationにのみ使用する。
4. QGISと目視による実用面の評価を継続する。
5. GSI-onlyでの改善が不十分な場合、または独立した最終精度の根拠が必要になった場合にのみmanual GTを検討する。

## Current Phase B work

最初に追加するteacher候補は **GSI Dataset-07 Water → OEM8 class 6 Water** である。

Water teacher audit:

- Image pairs: 1,250
- Positive-bearing images: 692
- All-ignore images: 558
- Positive pixels: 75,739,120
- Positive ratio: 18.519%
- Pairing: valid
- Size mismatch: documented repair後は0
- 配布された `val/554.png` は574x574であり、ローカル作業コピーで `[0:572, 0:572]` にcropして修復した。
- Remaining non-label mismatch: image 529の2 pixels
- Org exact label-color count: 0
- Original Base argmax on Water-positive pixels:
  - Water: 71.6103%
  - Agriculture: 5.6220%
- Mean Base Water probability: approximately 0.6697
- Mean Base Agriculture probability: approximately 0.0555

Paddy / Water datasets間でbyte-identicalなsource imageのSHA256 overlapは **0** である。

Phase B v0.1 / v0.2の実データPilotとSACLAJ development evaluationは完了した。
v0.1は`beta_water=1.0`、v0.2はWater source objective全体だけを0.5倍し、両者とも
1,028 logical steps / optimizer updates、Water 553 stepsで実行した。

- Original Baseから開始する。
- 既存v0.3のPaddy positive supervisionを維持する。
- 既存のPaddy unknown Base-KLを維持する。
- 既存のPaddy all-ignore replayを維持する。
- Water-positive supervisionを追加する。
- Water unknown pixelsではBase preservationを使用する。
- 最初のPilotではWater all-ignore imagesをreplayに追加しない。
- v0.3との比較可能性のため、optimizer updatesは1,028に保つ。
- v0.1は`beta_water=1.0`、v0.2はその他の条件を維持してWater source objective全体の
  weightだけを0.5にする。

主要結果は、v0.3 / v0.1 / v0.2の順にRice 87 / 83 / 85%、Other crop 61 / 53 / 57%、
Agriculture leakage 2.625 / 1.377 / 2.5%、Water 72 / 89 / 87%である。v0.2はWater改善を
かなり維持しつつAgriculture側を一部戻す妥協点候補だが、Tree系は明確に回復しなかった。
したがってWater単独の細かいweight tuningは一旦停止し、multi-teacher GSI-only modelへ進む。
v0.2を最適とは断定せず、v0.1も棄却しない。詳細は
[Phase B Water milestone](PHASE_B_WATER_MILESTONE_20260919.md)を参照すること。

## Evaluation status

SACLAJ 1,000 pointsはdevelopment evaluationであり、final holdoutではない。

Key v0.3 development metrics:

- Rice → Agriculture: 87%
- Other crop → Agriculture: 61%
- Agriculture leakage: 2.625%
- Water → Water: 72%

Base Waterは81%であり、Waterは未解決のriskとして残る。

## Manual GT policy

Manual GTは直近のフェーズではない。次のいずれかに該当する場合に検討する。

- GSI-onlyによる改善がplateauに達した場合。
- 実用上の品質が不十分なままの場合。
- 独立した最終精度またはproduction-readinessの根拠が必要になった場合。

## Current constraints / unresolved issues

- 一般的なaccuracy improvementは主張できない。
- 全国へのgeneralizationは主張できない。
- v0.3はproduction modelではない。
- SACLAJはfinal holdoutではない。
- Water改善とAgriculture/Tree保持のbalanceは未解決である。
- Water weightだけでTree degradationは解決せず、原因もWater weightだけに帰属できない。
- Base modelのlicensingおよびenterprise-use条件は未解決である。
- ローカルのrestricted dataはGitHubへ追加しない。
- 次はRoad / Building / Tree等のteacherを追加し、multi-teacher条件でWater weightを再評価する。
