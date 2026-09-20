cd /d C:\OpenEarthMap_PoC\openearthmap-poc

set "SACLAJ_CSV=C:\OpenEarthMap_PoC\data\saclaj\raw\Gref_DB_2025_06.csv"
set "MAPPING=C:\OpenEarthMap_PoC\data\saclaj\saclaj_mapping.v0.1.json"
set "BASE=C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
set "FT=C:\OpenEarthMap_PoC\openearthmap-poc\training_outputs\gsi_phase_b_v01\20260919T151035_437811Z_0fd54ade\checkpoints\best.pth"
set "RESULTS=C:\OpenEarthMap_PoC\data\saclaj\results"

python -m src.evaluation.evaluate_s
