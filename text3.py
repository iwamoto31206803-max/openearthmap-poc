python -m tools.evaluation.evaluate_gt54_predictions ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --prediction-dir C:\OpenEarthMap_PoC\runs\gt54_step4a_full\A\predictions ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_eval_A ^
  --model-id A

    python -m tools.evaluation.evaluate_gt54_predictions ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --prediction-dir C:\OpenEarthMap_PoC\runs\gt54_step4a_full\B\predictions ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_eval_B ^
  --model-id B

    python -m tools.evaluation.evaluate_gt54_predictions ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --prediction-dir C:\OpenEarthMap_PoC\runs\gt54_step4a_full\C\predictions ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_eval_C ^
  --model-id C

    python -m tools.evaluation.evaluate_gt54_predictions ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --prediction-dir C:\OpenEarthMap_PoC\runs\gt54_step4a_full\D\predictions ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_eval_D ^
  --model-id D
