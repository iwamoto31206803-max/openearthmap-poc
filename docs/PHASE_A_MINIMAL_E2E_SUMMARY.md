# Phase A Minimal End-to-End 実験記録

## 1. 位置づけと目的

本記録は、manual ground truth（manual GT）を作成しないPhase Aにおいて、
`teacher preparation → fine-tuning → SACLAJ development evaluation`の最小
End-to-End経路と、適応とBase出力保持のtrade-offを確認した実験のまとめである。
production modelの成果報告、production accuracyの証明、または最終的なモデル採用判断ではない。

GSI由来の水田partial labelはOEM8 class 7（Cropland / Agriculture）だけをpositiveとし、
その他をclass 255（unknown / ignore）とした。OEM8の既存class IDは変更していない。

## 2. Datasetと共通条件

GSI航空写真2,600画像のうちpositive-bearingは1,286枚、全画素unknownのFALSE画像は
1,314枚である。positive-bearing画像はseed 42でtrain 1,028枚 / validation 258枚に
決定的に分割した。unknownはBackgroundでもnegativeでもなく、このimage-level splitは
spatial independenceを保証しない。

Baseは`RGB_Real_5_u-efficientnet-b4.pth`、architectureは9-class SMP U-Net
（EfficientNet-B4 encoder、scSE）である。encoderとtrain時のBN/dropoutを固定し、decoderと
segmentation headだけを更新した。

## 3. 学習方法の変遷

- **v0.1 positive-only:** positive画素だけにcross entropyを適用した。Agriculture適応は強いが、
  Agriculture leakageとTree / Waterの崩壊を生じ、直接のpositive-only適応が危険と判断した。
- **v0.2 Base-preservation:** original Baseから再開始し、positive CEにunknown画素上の
  `KL(Base teacher || student)`を加えた。collapseを大幅に抑えたが、Water保持は34%に留まった。
- **v0.3 preservation-only replay:** v0.2の目的関数に、all-ignore/FALSE画像に対する
  Base-teacher preservation lossを追加した。FALSE画像をAgriculture negativeとして直接教師化せず、
  Base分布の保持だけに使うことが目的である。

## 4. v0.3会社PC Pilot結果

preflightはPASSし、画像数、split、Base SHA256、encoder固定、decoder / segmentation headの
trainable状態、teacherの`eval` / `no_grad`とoptimizer対象外を確認した。preflightで学習は
行っていない。smokeもPASSし、positive 1 batch + replay 1 batchでbackwardと
`optimizer.step()`が完了した。smoke-onlyの数値はperformance evidenceとして扱わない。

1 epoch Pilot（run ID `20260917T160534_801449Z_8c785b89`）は`completed`となった。
positive train / validationは1,028 / 258、replay train / validationは1,051 / 263、
`alpha_replay=1.0`である。詳細なloss、hashと実行provenanceは
[Replay-Preservation v0.3](REPLAY_PRESERVATION_V03.md)に記録する。

## 5. SACLAJ development evaluation

同一の固定development sample（10 subtype × 100 = 1,000点、seed 42）のaggregateを比較した。
値は期待OEM8 classとのpoint agreementである。Agriculture leakageは非Agriculture 800点で
Agricultureを予測した割合で、低い方向が望ましい。`~14%`はv0.1の既存記録の近似値である。

| Metric | Base | v0.1 | v0.2 | v0.3 |
|---|---:|---:|---:|---:|
| Rice → Agriculture | 27% | 99% | 84% | 87% |
| Other crop → Agriculture | 24% | 92% | 56% | 61% |
| Agriculture leakage | 3.375% | 39.25% | 6.5% | 2.625% |
| Broadleaf → Tree | 89% | 13% | 90% | 91% |
| Needleleaf → Tree | 83% | 7% | 89% | 87% |
| Mixed → Tree | 87% | 17% | 91% | 88% |
| Water → Water | 81% | 6% | 34% | 72% |
| Needleleaf evergreen → Tree | 88% | ~14% | 96% | 95% |
| Broadleaf evergreen → Tree | 88% | 15% | 88% | 90% |
| Needleleaf deciduous → Tree | 85% | 6% | 96% | 89% |
| Broadleaf deciduous → Tree | 79% | 6% | 85% | 81% |

v0.2からv0.3で、Water保持は34% → 72%、Riceは84% → 87%、Other cropは
56% → 61%、Agriculture leakageは6.5% → 2.625%となった。これは固定SACLAJ
development sample上で、v0.3がより好ましいadaptation-preservation trade-offを示した、
またはclass agreement / retentionが改善したという観察であり、accuracy improvementや
production readinessの主張ではない。

v0.1 / v0.2の結果が後続設計に影響したため、SACLAJはもはやfinal holdoutではなく
**development evaluation**である。上限付き層化標本は日本全体のclass頻度を示さず、
pixel-perfect GTでもない。training領域やBase pretrainingとの地理的重複の意味での統計的独立性は
主張せず、SACLAJ観測時期とGSI imageryのtemporal mismatchもあり得る。

## 6. v0.2 → v0.3評価入力同一性

会社PCローカルのread-only比較で、v0.2評価run
`20260916T135520_362028Z_4a047f8d`とv0.3評価run
`20260918T000738_565461Z_3d6b2f0b`の入力同一性を確認した。

- sampled site set / successful site set / subtype counts: **PASS**
- common successful sites: 1,000
- same / different / missing patch SHA256: 1,000 / 0 / 0
- Base argmax mismatches: 0
- Base expected-probability mismatches: 0
- Base Agriculture-probability mismatches: 0
- **V0.2 vs V0.3 EVALUATION INPUT IDENTITY: PASS**

v0.2とv0.3のSACLAJ development evaluationは同じsampled sitesとbyte-identicalな保存済み
GSI patchesを使った。したがって、v0.2–v0.3の指標差は同じ評価入力上の
model-output differencesとして扱える。この記録には地点ID、座標、地点別予測、patch名は含めない。

## 7. Phase A結論

Phase Aにより、direct positive-only adaptationは無関係なclassをcollapseさせ得るため
危険であることが分かった。Base-distribution preservationはこの失敗を大幅に抑え、
all-ignore/FALSE画像のpreservation-only replayを追加すると、固定SACLAJ development
sample上でadaptation-preservation trade-offがさらに改善した。したがってv0.3は現時点で
最も有力なPhase A学習方式candidateであるが、production modelではない。最終選定には、
manual GTまたは同等の独立評価データを用いたspatially/unseen evaluationが必要である。

## 8. Provenanceと情報管理

v0.3 PilotのBase SHA256は
`852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c`、実行コードの
git SHAは`23b1862cdda6ba30dedde9534e349f8c8c564d79`、best checkpoint SHA256は
`e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172`である。historical
v0.1 / v0.2 git SHAは未確認値を推測せず、記載しない。

実runのmanifest、checkpoint、training output、SACLAJ CSV、座標、地点ID、地点別結果、
patchは会社PCローカルにのみ保持し、Gitに含めない。本記録には公開可能な実行識別子、
hash、件数とaggregate指標のみを記録する。

実装と個別手順は、[Phase A v0.1](GSI_PHASE_A_TRAINING.md)、
[Base-Preservation v0.2](BASE_PRESERVATION_FINETUNING.md)、
[Replay-Preservation v0.3](REPLAY_PRESERVATION_V03.md)、
[SACLAJ Evaluation](SACLAJ_EVALUATION.md)を参照すること。
