python -c "from pathlib import Path; import numpy as np; from PIL import Image; root=Path(r'C:\OpenEarthMap_PoC\data\gsi\raw\road_572'); total=red=diff=diff_not_red=org_red=0; orgs={p.relative_to(root/'org'):p for p in (root/'org').rglob('*.png')}; vals={p.relative_to(root/'val'):p for p in (root/'val').rglob('*.png')}; exec('for k in sorted(orgs):\n o=np.asarray(Image.open(orgs[k]).convert(\"RGB\")); v=np.asarray(Image.open(vals[k]).convert(\"RGB\")); r=np.all(v==[255,0,0],axis=2); d=np.any(v!=o,axis=2); total+=d.size; red+=int(r.sum()); diff+=int(d.sum()); diff_not_red+=int((d & ~r).sum()); org_red+=int(np.all(o==[255,0,0],axis=2).sum())'); print('total pixels =',total); print('val exact red =',red); print('different pixels =',diff); print('different but val not red =',diff_not_red); print('org exact red =',org_red)"

python -m src.training.prepare_gsi_labels ^
  C:\OpenEarthMap_PoC\data\gsi\raw\road_572 ^
  C:\OpenEarthMap_PoC\data\gsi\prepared\road_572 ^
  --gsi-category road ^
  --oem-class-id 4 ^
  --label-color 255,0,0
