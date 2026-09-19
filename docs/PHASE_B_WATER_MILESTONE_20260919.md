# Phase B Water-positive supervision milestone (2026-09-19)

## Scope and experiment design

Phase BはOriginal Baseから開始し、Paddy positive supervision、Water positive supervision、
Paddy all-ignore replayを使う。Water all-ignoreは使用しない。v0.1の`beta_water=1.0`に対し、
v0.2は`beta_water=0.5`のみを変更し、Water positive CEとWater unknown Base-preservation KLを
含むWater source objective全体に0.5を掛けた。両実験とも1,028 logical steps / optimizer
updates、Water 553 stepsである。

## Results

SACLAJ development evaluationの集計値は次のとおりである。SACLAJはmanual GT、final
holdout、production accuracyの評価ではない。

| Metric | Phase A v0.3 | Phase B v0.1 | Phase B v0.2 |
|---|---:|---:|---:|
| Rice | 87% | 83% | 85% |
| Other crop | 61% | 53% | 57% |
| Agriculture leakage | 2.625% | 1.377% | 2.5% |
| Water | 72% | 89% | 87% |
| Broadleaf | — | 88% | 88% |
| Needleleaf | — | 84% | 84% |
| Mixed | — | 85% | 85% |
| Needleleaf evergreen | — | 88% | 88% |
| Broadleaf evergreen | — | 85% | 81% |
| Needleleaf deciduous | — | about 85.9% (n=99, 1 error) | 83% |
| Broadleaf deciduous | — | 78% | 80% |

v0.2のWater mean expected probabilityはBase 0.759859742...から0.681334242...へ低下し、
Water mean Agriculture probabilityはBase 0.008055887...から0.132846135...へ上昇した。
その他のTree subtypeでもexpected Tree probability低下とAgriculture probability上昇が継続した。

## Input identity verification

v0.1とPhase A v0.3の確認はsampled IDs 1,000/1,000一致、common-success 999であり、
patch SHA256、Base predicted class、Base expected probability、Base Agriculture probabilityの
mismatchはすべて0であった。

Phase A v0.3 run `20260918T000738_565461Z_3d6b2f0b`とPhase B v0.2 evaluation run
`20260919T101351_262021Z_48be0033`は **PERFECT IDENTITY** であった。

- sampled rows: 1,000 vs 1,000; common IDs: 1,000; only v0.3 / only v0.2: 0 / 0
- subtype / expected-class / status mismatches: 0 / 0 / 0
- success: v0.3 1,000; v0.2 1,000
- patch SHA256 / Base predicted-class / Base expected-probability / Base Agriculture-probability mismatches:
  0 / 0 / 0 / 0

したがって、この1,000地点におけるv0.3とv0.2のSACLAJ差はinput差ではなくmodel差として扱える。

## Interpretation and decision

`beta_water`を1.0から0.5に下げるとWaterは89%から87%へ小幅に低下したが、Riceは
83%から85%、Other cropは53%から57%へ戻った。Agriculture leakageは1.377%から2.5%へ戻り、
v0.3の2.625%に近づいた。一方、Tree系は明確に回復せず、一部subtypeはむしろ悪化した。

これにより、**Water supervisionの強さ調整だけではTree degradationは解決しない**と判断する。
Water単独の細かいweight tuningは一旦停止する。v0.2はWater改善をかなり維持しつつ
Agriculture側を一部戻す妥協点候補として保持するが、最適とは断定しない。v0.1も棄却せず、
Water単独の最適化が完了したともしない。Tree degradationの原因をWater weightだけに帰属しない。

## Unresolved issues and next direction

- Water改善とAgriculture / Tree保持のbalanceは未解決である。
- Tree subtypeのexpected probability低下とAgriculture probability上昇の原因は未解決である。
- 本開発評価だけからnationwide accuracy、production readiness、generalizationを主張できない。

次はRoad / Building / Tree等のteacherを追加したmulti-teacher GSI-only modelへ進み、その条件下で
Water weightを再評価する。
