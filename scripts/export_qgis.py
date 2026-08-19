"""Assemble a QGIS-ready package: RGBI imagery + TESSERA change outputs per site.

Writes outputs/qgis/ with, per site:
  - before/after RGBI GeoTIFFs (copied from data/, QGIS-friendly names)
  - tessera_cosdist GeoTIFF (continuous change strength)
  - tessera_change_binary GeoTIFF (Otsu-thresholded mask)
  - AOI geojson
  - sidecar .qml styles so every layer loads pre-styled
Then zips the folder.
"""

import glob
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import numpy as np
import rasterio

OUT = Path("outputs/qgis")

AOI_FILES = {
    "chiba": "Site1_Tomisato_Chiba_AOI.geojson",
    "bassin": "Bassin_AOI.geojson",
}
TESSERA_YEARS = {"chiba": (2020, 2024), "bassin": (2017, 2025)}


def qml_rgb(img_path):
    """Multiband color QML with baked 2-98% per-band stretch (bands 3,2,1 = RGB)."""
    with rasterio.open(img_path) as src:
        arr = src.read()
    lims = []
    for b in [2, 1, 0]:  # red, green, blue band indices in the stack
        v = arr[b][arr[b] > 0]
        lo, hi = np.percentile(v, [2, 98])
        lims.append((int(lo), int(hi)))
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
 <pipe>
  <rasterrenderer type="multibandcolor" redBand="3" greenBand="2" blueBand="1" opacity="1">
   <redContrastEnhancement><minValue>{lims[0][0]}</minValue><maxValue>{lims[0][1]}</maxValue><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>
   <greenContrastEnhancement><minValue>{lims[1][0]}</minValue><maxValue>{lims[1][1]}</maxValue><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>
   <blueContrastEnhancement><minValue>{lims[2][0]}</minValue><maxValue>{lims[2][1]}</maxValue><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>
  </rasterrenderer>
 </pipe>
</qgis>
"""


def qml_cosdist(vmax):
    """Singleband pseudocolor QML, magma-like ramp from 0 to p99.5."""
    stops = [
        (0.00, "000004"), (0.25, "51127c"), (0.50, "b73779"),
        (0.75, "fc8961"), (1.00, "fcfdbf"),
    ]
    items = "".join(
        f'<item alpha="255" value="{v*vmax:.4f}" color="#{c}" label="{v*vmax:.2f}"/>'
        for v, c in stops
    )
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
 <pipe>
  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" classificationMin="0" classificationMax="{vmax:.4f}">
   <rastershader>
    <colorrampshader colorRampType="INTERPOLATED" clip="0">{items}</colorrampshader>
   </rastershader>
  </rasterrenderer>
 </pipe>
</qgis>
"""


QML_BINARY = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
 <pipe>
  <rasterrenderer type="paletted" band="1" opacity="0.8">
   <colorPalette>
    <paletteEntry value="0" color="#00000000" alpha="0" label="no change"/>
    <paletteEntry value="1" color="#ff2b55" alpha="255" label="changed"/>
   </colorPalette>
  </rasterrenderer>
 </pipe>
</qgis>
"""


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for site in ["chiba", "bassin"]:
        y1, y2 = TESSERA_YEARS[site]
        # 1. imagery
        for epoch in ["before", "after"]:
            src_path = sorted(glob.glob(f"data/{site}_{epoch}_*.tif"))[0]
            date = Path(src_path).stem.split("_")[-1]
            dst = OUT / f"{site}_S2_{epoch}_{date}_RGBI.tif"
            shutil.copy(src_path, dst)
            dst.with_suffix(".qml").write_text(qml_rgb(src_path))

        # 2. TESSERA continuous change map
        cd_src = f"outputs/{site}_tessera_cosdist.tif"
        dst = OUT / f"{site}_TESSERA_cosine_distance_{y1}_{y2}.tif"
        shutil.copy(cd_src, dst)
        with rasterio.open(cd_src) as s:
            vmax = float(np.percentile(s.read(1), 99.5))
        dst.with_suffix(".qml").write_text(qml_cosdist(vmax))

        # 3. TESSERA binary mask
        t = np.load(f"outputs/{site}_tessera.npz")
        with rasterio.open(cd_src) as s:
            profile = s.profile
        profile.update(dtype="uint8", nodata=None)
        dst = OUT / f"{site}_TESSERA_change_binary_{y1}_{y2}.tif"
        with rasterio.open(dst, "w", **profile) as d:
            d.write(t["tess_bin"].astype("uint8"), 1)
        dst.with_suffix(".qml").write_text(QML_BINARY)

        # 4. AOI
        shutil.copy(AOI_FILES[site], OUT / f"{site}_AOI.geojson")

    readme = OUT / "README.txt"
    readme.write_text(
        "QGIS package - Sentinel-2 imagery + TESSERA change detection\n"
        "=============================================================\n\n"
        "Per site (chiba = Shimizu-Tomisato JP, bassin = SNCF Bassin FR):\n"
        "  *_S2_before/after_<date>_RGBI.tif   4-band Sentinel-2 L2A (blue,green,red,nir), 10 m,\n"
        "                                      native UTM (chiba EPSG:32654, bassin EPSG:32631)\n"
        "  *_TESSERA_cosine_distance_*.tif     continuous change strength (1 - cosine similarity\n"
        "                                      of 128-d annual TESSERA embeddings), 0=identical\n"
        "  *_TESSERA_change_binary_*.tif       Otsu-thresholded mask (1=changed), drape over the\n"
        "                                      after-image\n"
        "  *_AOI.geojson                       site AOI outline\n\n"
        "All rasters of a site share the same 10 m grid (pixel-aligned).\n"
        ".qml sidecar files are included: QGIS applies them automatically on layer load\n"
        "(true-color 2-98% stretch for imagery, magma ramp for cosine distance,\n"
        "transparent/red palette for the binary mask).\n\n"
        "TESSERA years: chiba 2020->2024 (JP 2025 embeddings not yet published),\n"
        "bassin 2017->2025 (earliest/latest available).\n"
    )

    shutil.make_archive("outputs/qgis_package", "zip", OUT)
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT}/ ({total/1e6:.1f} MB) and outputs/qgis_package.zip")
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)
