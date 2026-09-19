python -m src.training.diagnose_gsi_overlay ^
  C:\OpenEarthMap_PoC\data\gsi\raw\Road_572 ^
  C:\OpenEarthMap_PoC\data\gsi\diagnostics\Road_572


python -c "from pathlib import Path; from PIL import Image; root=Path(r'C:\OpenEarthMap_PoC\data\gsi\raw\Road_572'); org={p.relative_to(root/'org'):p for p in (root/'org').rglob('*.png')}; val={p.relative_to(root/'val'):p for p in (root/'val').rglob('*.png')}; common=sorted(set(org)&set(val)); missing_org=sorted(set(val)-set(org)); missing_val=sorted(set(org)-set(val)); mism=[]; [mism.append((str(k),Image.open(org[k]).size,Image.open(val[k]).size)) for k in common if Image.open(org[k]).size!=Image.open(val[k]).size]; print('org =',len(org)); print('val =',len(val)); print('common =',len(common)); print('missing org =',len(missing_org)); print('missing val =',len(missing_val)); print('size mismatches =',len(mism)); print('first mismatches =',mism[:20])"

python -m src.training.prepare_gsi_labels ...
