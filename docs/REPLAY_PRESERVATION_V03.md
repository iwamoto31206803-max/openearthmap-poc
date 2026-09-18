# Replay-Preservation Fine-tuning v0.3

## 位置づけ

v0.3はv0.2 Base-preservationに、all-ignore/FALSE画像のBase-teacher出力を保持する
replay lossを追加したPhase A Pilotである。FALSE画像をclass 7のnegativeとして学習するのではなく、
preservation-onlyで使う。OEM8 class IDと既存の地理空間条件は変更しない。

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
