cd /d C:\OpenEarthMap_PoC\openearthmap-poc

python compare_base_ft.py ^
  --lat 35.2986808 ^
  --lon 139.1314632 ^
  --name karigawa01 ^
  --base-model C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth ^
  --fine-tuned-model C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T151035_437811Z_0fd54ade\checkpoints\best.pth ^
  --output-dir C:\OpenEarthMap_PoC\comparison\phase_b_v03_road


python compare_teacher_progression.py ^
  --input C:\OpenEarthMap_PoC\comparison\phase_b_v03_road\karigawa01\input\gsi_rgb.tif ^
  --name karigawa01 ^
  --base-model C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth ^
  --paddy-model C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a_v03\pilot\20260917T160534_801449Z_8c785b89\checkpoints\best.pth ^
  --water-model C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T053943_488504Z_000833f3\checkpoints\best.pth ^
  --road-model C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T151035_437811Z_0fd54ade\checkpoints\best.pth ^
  --output-dir C:\OpenEarthMap_PoC\teacher_progression ^
  --focus-transition 65 ^
  --focus-transition 56 ^
  --focus-transition 61 ^
  --focus-transition 16 ^
  --focus-transition 74



  python compare_classification_progression.py ^
  --stage-00 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\00_base\classes_sieve_5m2.tif ^
  --stage-01 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\01_paddy\classes_sieve_5m2.tif ^
  --stage-02 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\02_paddy_water\classes_sieve_5m2.tif ^
  --stage-03 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\03_paddy_water_road\classes_sieve_5m2.tif ^
  --output-dir C:\OpenEarthMap_PoC\transition_diagnostics\karigawa01 ^
  --changed-geotiff
  
