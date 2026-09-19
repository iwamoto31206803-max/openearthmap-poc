set "SACLAJ_CSV=C:\OpenEarthMap_PoC\data\saclaj\raw\Gref_DB_2025_06.csv"
set "MAPPING=C:\OpenEarthMap_PoC\data\saclaj\saclaj_mapping.v0.1.json"
set "BASE=C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
set "FT=C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260918T163820_370416Z_a99fb364\checkpoints\best.pth"
set "RESULTS=C:\OpenEarthMap_PoC\data\saclaj\results"

python -m src.evaluation.evaluate_saclaj ^
  --saclaj-csv "%SACLAJ_CSV%" --mapping "%MAPPING%" ^
  --base-model "%BASE%" --fine-tuned-model "%FT%" ^
  --output-dir "%RESULTS%" --device cpu --num-threads 2 ^
  --max-samples-per-category 100 --seed 42 --preflight
