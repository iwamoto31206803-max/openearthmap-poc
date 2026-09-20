python compare_classification_progression.py ^
  --stage-00 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\00_base\classes_sieve_5m2.tif ^
  --stage-01 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\01_paddy\classes_sieve_5m2.tif ^
  --stage-02 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\02_paddy_water\classes_sieve_5m2.tif ^
  --stage-03 C:\OpenEarthMap_PoC\teacher_progression\karigawa01\stages\03_paddy_water_road\classes_sieve_5m2.tif ^
  --output-dir C:\OpenEarthMap_PoC\transition_diagnostics\karigawa01 ^
  --changed-geotiff


python -c "import json,pathlib; root=pathlib.Path(r'C:\OpenEarthMap_PoC\transition_diagnostics\karigawa01'); pairs={(3,6),(4,6),(1,6),(2,6),(6,3),(6,4),(6,1),(6,2),(3,4),(4,3),(2,4),(7,4)}; exec(\"for n in ['00_to_01','01_to_02','02_to_03']:\n d=json.loads((root/n/'summary.json').read_text(encoding='utf-8'))\n print('\\n===',n,'===')\n print('changed=',d['changed_pixel_count'],f\\\"({d['changed_percent']:.4f}%)\\\")\n [print(f\\\"{x['from_id']}->{x['to_id']} {x['from_name']} -> {x['to_name']}: {x['pixel_count']} px, {x['percent_of_all']:.4f}% all, {x['percent_of_changed']:.4f}% changed\\\") for x in d['transitions'] if (x['from_id'],x['to_id']) in pairs]\")"
