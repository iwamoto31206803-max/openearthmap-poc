python -m tools.evaluation.compare_gt54_models ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --inference-root C:\OpenEarthMap_PoC\runs\gt54_step4a_full ^
  --ownership-dir C:\OpenEarthMap_PoC\runs\gt54_step1_ownership ^
  --output-dir C:\OpenEarthMap_PoC\runs\gt54_step4b_comparison
