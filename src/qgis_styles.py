from __future__ import annotations

from pathlib import Path
import argparse
from html import escape

if __package__:
    from .config import CLASS_COLORS, CLASS_NAMES
else:
    from config import CLASS_COLORS, CLASS_NAMES


# Values written by compare_base_ft.py.  Zero is deliberately transparent so
# that the five road-related transitions stand out over the source imagery.
ROAD_CHANGE_STYLES = {
    0: ("Other / not focused", (0, 0, 0, 0)),
    1: ("Pavement -> Road", (0, 170, 255, 255)),
    2: ("Road -> Road", (255, 255, 255, 255)),
    3: ("Road -> Pavement", (255, 170, 0, 255)),
    4: ("Other -> Road", (0, 220, 80, 255)),
    5: ("Road -> Other", (230, 40, 60, 255)),
}


def hex_color(rgba: tuple[int, int, int, int]) -> str:
    r, g, b, _ = rgba
    return f"#{r:02x}{g:02x}{b:02x}"


def write_class_raster_style(raster_path: Path, opacity: float = 0.5) -> Path:
    style_path = raster_path.with_suffix(".qml")

    entries = []
    for class_id, name in CLASS_NAMES.items():
        entries.append(
            f'          <paletteEntry value="{class_id}" '
            f'color="{hex_color(CLASS_COLORS[class_id])}" '
            f'alpha="255" label="{escape(name)}"/>'
        )

    qml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.40" styleCategories="Symbology">\n'
        "  <pipe>\n"
        '    <provider>\n'
        '      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" '
        'zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/>\n'
        "    </provider>\n"
        f'    <rasterrenderer type="paletted" band="1" opacity="{opacity:.6f}" '
        'alphaBand="-1" nodataColor="">\n'
        "      <rasterTransparency/>\n"
        "      <colorPalette>\n"
        + "\n".join(entries)
        + "\n      </colorPalette>\n"
        "    </rasterrenderer>\n"
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
        '    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" '
        'colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0"/>\n'
        '    <rasterresampler maxOversampling="2"/>\n'
        "  </pipe>\n"
        "  <blendMode>0</blendMode>\n"
        "</qgis>\n"
    )
    style_path.write_text(qml, encoding="utf-8")
    return style_path


def write_confidence_style(raster_path: Path) -> Path:
    style_path = raster_path.with_suffix(".qml")

    qml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.40" styleCategories="Symbology">\n'
        "  <pipe>\n"
        "    <provider>\n"
        '      <resampling enabled="false" zoomedInResamplingMethod="bilinear" '
        'zoomedOutResamplingMethod="bilinear" maxOversampling="2"/>\n'
        "    </provider>\n"
        '    <rasterrenderer type="singlebandgray" grayBand="1" opacity="1" '
        'alphaBand="-1" gradient="BlackToWhite" nodataColor="">\n'
        "      <rasterTransparency/>\n"
        "      <minMaxOrigin>\n"
        "        <limits>None</limits>\n"
        "        <extent>WholeRaster</extent>\n"
        "        <statAccuracy>Estimated</statAccuracy>\n"
        "        <cumulativeCutLower>0.02</cumulativeCutLower>\n"
        "        <cumulativeCutUpper>0.98</cumulativeCutUpper>\n"
        "        <stdDevFactor>2</stdDevFactor>\n"
        "      </minMaxOrigin>\n"
        "      <contrastEnhancement>\n"
        "        <minValue>0</minValue>\n"
        "        <maxValue>1</maxValue>\n"
        "        <algorithm>StretchToMinimumMaximum</algorithm>\n"
        "      </contrastEnhancement>\n"
        "    </rasterrenderer>\n"
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
        '    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" '
        'colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0"/>\n'
        '    <rasterresampler maxOversampling="2"/>\n'
        "  </pipe>\n"
        "  <blendMode>0</blendMode>\n"
        "</qgis>\n"
    )
    style_path.write_text(qml, encoding="utf-8")
    return style_path


def write_road_change_style(raster_path: Path, opacity: float = 0.85) -> Path:
    """Write a categorical QGIS style for the road-focused comparison map."""
    style_path = raster_path.with_suffix(".qml")
    entries = []
    for value, (label, rgba) in ROAD_CHANGE_STYLES.items():
        r, g, b, alpha = rgba
        entries.append(
            f'          <paletteEntry value="{value}" color="#{r:02x}{g:02x}{b:02x}" '
            f'alpha="{alpha}" label="{escape(label)}"/>'
        )
    qml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.40" styleCategories="Symbology">\n'
        "  <pipe>\n"
        f'    <rasterrenderer type="paletted" band="1" opacity="{opacity:.6f}" '
        'alphaBand="-1" nodataColor="">\n'
        "      <rasterTransparency/>\n      <colorPalette>\n"
        + "\n".join(entries)
        + "\n      </colorPalette>\n    </rasterrenderer>\n"
        "  </pipe>\n  <blendMode>0</blendMode>\n</qgis>\n"
    )
    style_path.write_text(qml, encoding="utf-8")
    return style_path


def write_vector_landcover_style(gpkg_path: Path, opacity: float = 0.5) -> Path:
    style_path = gpkg_path.with_suffix(".qml")

    categories = []
    symbols = []

    for symbol_id, (class_id, name) in enumerate(CLASS_NAMES.items()):
        categories.append(
            f'      <category value="{class_id}" symbol="{symbol_id}" '
            f'label="{escape(name)}" render="true"/>'
        )
        r, g, b, _ = CLASS_COLORS[class_id]
        symbols.append(
            f'''      <symbol type="fill" name="{symbol_id}" alpha="{opacity:.6f}" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="{r},{g},{b},255"/>
            <Option name="outline_color" type="QString" value="70,70,70,180"/>
            <Option name="outline_style" type="QString" value="solid"/>
            <Option name="outline_width" type="QString" value="0.15"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>'''
        )

    qml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.40" styleCategories="Symbology">\n'
        '  <renderer-v2 type="categorizedSymbol" attr="class_id" '
        'enableorderby="0" forceraster="0" symbollevels="0">\n'
        "    <categories>\n"
        + "\n".join(categories)
        + "\n    </categories>\n"
        "    <symbols>\n"
        + "\n".join(symbols)
        + "\n    </symbols>\n"
        "  </renderer-v2>\n"
        "  <blendMode>0</blendMode>\n"
        "  <featureBlendMode>0</featureBlendMode>\n"
        "  <layerGeometryType>2</layerGeometryType>\n"
        "</qgis>\n"
    )
    style_path.write_text(qml, encoding="utf-8")
    return style_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sidecar QGIS .qml styles for PoC outputs."
    )
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--confidence", nargs="*", default=[])
    parser.add_argument("--vectors", nargs="*", default=[])
    parser.add_argument("--road-change", nargs="*", default=[])
    parser.add_argument(
        "--landcover-opacity",
        type=float,
        default=0.5,
        help="Land-cover opacity in QGIS, 0-1. Default: 0.5",
    )
    args = parser.parse_args()

    if not 0 <= args.landcover_opacity <= 1:
        raise ValueError("--landcover-opacity must be between 0 and 1.")

    created: list[Path] = []

    for value in args.classes:
        path = Path(value)
        if path.exists():
            created.append(write_class_raster_style(path, args.landcover_opacity))

    for value in args.confidence:
        path = Path(value)
        if path.exists():
            created.append(write_confidence_style(path))

    for value in args.vectors:
        path = Path(value)
        if path.exists():
            created.append(write_vector_landcover_style(path, args.landcover_opacity))

    for value in args.road_change:
        path = Path(value)
        if path.exists():
            created.append(write_road_change_style(path))

    print()
    print("QGIS STYLES OK")
    print("-------------------------------------")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
