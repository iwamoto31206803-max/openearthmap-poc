mkdir C:\OpenEarthMap_PoC\runs\gt54_perfect_predictions

copy C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54\labels\*.tif C:\OpenEarthMap_PoC\runs\gt54_perfect_predictions\

python -m tools.evaluation.evaluate_gt54_predictions ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --prediction-dir C:\OpenEarthMap_PoC\runs\gt54_perfect_predictions ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_step3_perfect ^
  --model-id PERFECT
