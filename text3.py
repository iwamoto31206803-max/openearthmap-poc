python -m tools.evaluation.predict_gt54_models ^
  --manifest manifests\val_gt_georef.csv ^
  --dataset-root C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54 ^
  --model-a C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth ^
  --model-b C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a_v03\pilot\20260917T160534_801449Z_8c785b89\checkpoints\best.pth ^
  --model-c C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T053943_488504Z_000833f3\checkpoints\best.pth ^
  --model-d C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T151035_437811Z_0fd54ade\checkpoints\best.pth ^
  --output-root C:\OpenEarthMap_PoC\runs\gt54_step4a_smoke ^
  --smoke-items 1
