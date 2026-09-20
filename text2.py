python compare_teacher_progression.py ^
  --input C:\OpenEarthMap_PoC\comparison\phase_b_v03_road\chiba01\input\gsi_rgb.tif ^
  --name chiba01 ^
  --base-model C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth ^
  --paddy-model C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a_v03\pilot\20260917T160534_801449Z_8c785b89\checkpoints\best.pth ^
  --water-model C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T053943_488504Z_000833f3\checkpoints\best.pth ^
  --road-model C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T151035_437811Z_0fd54ade\checkpoints\best.pth ^
  --output-dir C:\OpenEarthMap_PoC\teacher_progression ^
  --focus-transition 85 ^
  --focus-transition 34
