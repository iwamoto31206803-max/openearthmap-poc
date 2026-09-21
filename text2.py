cd /d C:\OpenEarthMap_PoC\oemsar_data

python -c "import rasterio,glob,os,csv; from rasterio.warp import transform_bounds; files=sorted(glob.glob(r'trainval\val\sar_images\*.tif')); rows=[]; \
[(lambda p: (lambda ds: (lambda b: rows.append([os.path.basename(p),str(ds.crs),(b.left+b.right)/2,(b.bottom+b.top)/2]))(transform_bounds(ds.crs,'EPSG:4326',*ds.bounds)))(rasterio.open(p)))(p) for p in files]; \
[(r.append('Japan' if 122<=r[2]<=154 and 20<=r[3]<=46 else 'France' if -6<=r[2]<=10 and 41<=r[3]<=52 else 'USA' if -130<=r[2]<=-60 and 20<=r[3]<=55 else 'Other')) for r in rows]; \
f=open('val_geography.csv','w',newline='',encoding='utf-8'); w=csv.writer(f); w.writerow(['file','crs','lon','lat','country']); w.writerows(rows); f.close(); \
from collections import Counter; print(Counter(r[4] for r in rows)); print('--- Japan ---'); [print(r) for r in rows if r[4]=='Japan']"
