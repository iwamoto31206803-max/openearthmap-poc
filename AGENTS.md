# Repository Guidelines

## Boundaries

- 標準実行環境には Python 3.11 を使用する。
- 依頼された変更に関連する文書・コードだけを確認する。
- 変更に関連するテストと構文チェックを実行し、必要な検証が通った後は検証範囲を不必要に広げない。
- OEM8 のクラス ID と既存の地理空間情報を維持する。
- モデル重み、学習データ、GeoTIFF、GeoPackage、チェックポイント、実行結果を Git に追加しない。
- SACLAJ の CSV、座標、地点別結果、パッチ、チェックポイント、ローカル実行結果を GitHub や外部環境へ追加しない。
- 変更は小さく保ち、無関係なファイルを編集しない。

## Routing

- Current project status and next priorities: `docs/CURRENT_STATUS.md`
- GSI teacher preparation: `docs/GSI_TRAINING_DATA.md`
- Phase A training: `docs/GSI_PHASE_A_TRAINING.md`
- Base preservation: `docs/BASE_PRESERVATION_FINETUNING.md`
- Replay preservation: `docs/REPLAY_PRESERVATION_V03.md`
- SACLAJ development evaluation: `docs/SACLAJ_EVALUATION.md`
- Phase A experiment record: `docs/PHASE_A_MINIMAL_E2E_SUMMARY.md`
