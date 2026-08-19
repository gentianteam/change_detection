"""Select season-matched, low-cloud scene pairs for both AOIs and download RGBI stacks."""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import geopandas as gpd
from sentinel2_stac import search_scenes, download_rgbi

DATA_DIR = "data"

SITES = {
    "chiba": {
        "aoi": "Site1_Tomisato_Chiba_AOI.geojson",
        "buffer_m": 100,
        # season-matched late-spring windows, 2020 vs 2025
        "windows": {
            "before": ("2020-04-15", "2020-06-30"),
            "after": ("2025-04-15", "2025-06-30"),
        },
    },
    "bassin": {
        "aoi": "Bassin_AOI.geojson",
        "buffer_m": 500,
        # earliest usable S2 (winter 2016) vs matching winter 2025/26
        "windows": {
            "before": ("2016-10-01", "2017-02-28"),
            "after": ("2025-11-01", "2026-02-28"),
        },
    },
}


def pick_best(gdf, max_nodata=15):
    g = gdf[gdf["nodata_pct"].fillna(0) < max_nodata]
    if not len(g):
        g = gdf
    return g.sort_values("cloud_cover").iloc[0]


if __name__ == "__main__":
    import os

    os.makedirs(DATA_DIR, exist_ok=True)
    for site, cfg in SITES.items():
        aoi = gpd.read_file(cfg["aoi"])
        for epoch, (start, end) in cfg["windows"].items():
            g = search_scenes(aoi, start, end, max_cloud=10)
            best = pick_best(g)
            out = f"{DATA_DIR}/{site}_{epoch}_{best['datetime'][:10]}.tif"
            print(
                f"{site} {epoch}: {best['id']}  cloud={best['cloud_cover']:.2f}% "
                f"nodata={best['nodata_pct']:.1f}%  -> {out}"
            )
            download_rgbi(best["item"], aoi, out, buffer_m=cfg["buffer_m"])
