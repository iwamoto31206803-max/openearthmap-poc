from pathlib import Path
from qgis.core import QgsProject, QgsRasterLayer

# -------------------------
# 設定
# -------------------------
RAS_DIR = Path(r"C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef")

# すでに読み込み済みで、見た目が整っているレイヤ名
TEMPLATE_LAYER_NAME = "ValArea_011_gt_georef"

GROUP_NAME = "val_gt_georef_all"
OPACITY = 0.45   # 透過率（0〜1）。航空写真を見やすくするなら 0.35〜0.5 くらいがおすすめ

project = QgsProject.instance()

# テンプレート取得
template_layers = project.mapLayersByName(TEMPLATE_LAYER_NAME)
if not template_layers:
    raise Exception(f"テンプレートレイヤが見つかりません: {TEMPLATE_LAYER_NAME}")

template = template_layers[0]

# グループ取得 or 作成
root = project.layerTreeRoot()
group = root.findGroup(GROUP_NAME)
if group is None:
    group = root.addGroup(GROUP_NAME)

# 既存ソース一覧
loaded_sources = set()
for lyr in project.mapLayers().values():
    try:
        loaded_sources.add(lyr.source())
    except:
        pass

# 一括ロード
ok = 0
skip = 0
ng = 0

for tif in sorted(RAS_DIR.glob("*_gt_georef.tif")):
    tif_str = str(tif)

    if tif_str in loaded_sources:
        print(f"[SKIP already loaded] {tif.name}")
        skip += 1
        continue

    layer = QgsRasterLayer(tif_str, tif.stem)
    if not layer.isValid():
        print(f"[NG invalid] {tif.name}")
        ng += 1
        continue

    # スタイルをテンプレートから複製
    layer.setRenderer(template.renderer().clone())
    layer.renderer().setOpacity(OPACITY)
    layer.triggerRepaint()

    project.addMapLayer(layer, False)
    group.addLayer(layer)

    print(f"[OK] {tif.name}")
    ok += 1

print("---- done ----")
print(f"OK   : {ok}")
print(f"SKIP : {skip}")
print(f"NG   : {ng}")
