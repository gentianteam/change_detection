"""Download the full monitoring scene series (RGBI + SCL cloud mask) per site.

Baseline years feed the monthly composites; the monitor window is what the
alerting loop replays. Each scene is stored as one .npz on the site's common
10 m grid (matching the pair-analysis grid).
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from sentinel2_stac import search_scenes, RGBI_ASSETS

SITES = {
    "chiba": {
        "aoi": "Site1_Tomisato_Chiba_AOI.geojson",
        "buffer_m": 100,
        "baseline": ("2022-01-01", "2024-12-31"),
        "monitor": ("2025-01-01", "2025-12-31"),
    },
    "bassin": {
        "aoi": "Bassin_AOI.geojson",
        "buffer_m": 500,
        "baseline": ("2022-08-01", "2025-07-31"),
        "monitor": ("2025-08-01", "2026-07-31"),
    },
}
MAX_CLOUD_BASELINE = 20
MAX_CLOUD_MONITOR = 40  # SCL masking handles partial cloud
MAX_BASELINE_PER_MONTH_YEAR = 1  # best scene per (year, month) keeps volume sane


def grid_for(site):
    """Reuse the exact grid of the pair-analysis rasters."""
    import glob

    path = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
    with rasterio.open(path) as src:
        return src.crs, src.transform, src.height, src.width


def fetch_scene(item, crs, transform, h, w):
    """Read RGBI (10 m) + SCL (20 m, upsampled) windows for the site grid."""
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    bands = np.zeros((4, h, w), dtype=np.uint16)
    for i, key in enumerate(RGBI_ASSETS):
        with rasterio.open(item.assets[key].href) as src:
            assert str(src.crs) == str(crs), f"tile CRS mismatch {src.crs}"
            win = from_bounds(left, bottom, right, top, src.transform)
            bands[i] = src.read(
                1, window=win.round_offsets().round_lengths(),
                boundless=True, fill_value=0,
            )[:h, :w]
    with rasterio.open(item.assets["scl"].href) as src:
        win = from_bounds(left, bottom, right, top, src.transform)
        scl20 = src.read(
            1, window=win.round_offsets().round_lengths(),
            boundless=True, fill_value=0,
        )
    scl = np.kron(scl20, np.ones((2, 2), dtype=np.uint8))[:h, :w]
    if scl.shape != (h, w):  # pad if the 20 m window came up one pixel short
        pad = np.zeros((h, w), dtype=np.uint8)
        pad[: scl.shape[0], : scl.shape[1]] = scl
        scl = pad
    return bands, scl


def dedupe_by_date(gdf):
    """Keep one item per acquisition date (lowest cloud)."""
    gdf = gdf.copy()
    gdf["date"] = gdf["datetime"].str[:10]
    return gdf.sort_values("cloud_cover").groupby("date").first().reset_index()


if __name__ == "__main__":
    for site, cfg in SITES.items():
        out_dir = f"data/monitor/{site}"
        os.makedirs(out_dir, exist_ok=True)
        aoi = gpd.read_file(cfg["aoi"])
        crs, transform, h, w = grid_for(site)

        # --- baseline: best scene per (year, month)
        g = search_scenes(aoi, *cfg["baseline"], max_cloud=MAX_CLOUD_BASELINE)
        g = dedupe_by_date(g)
        g["ym"] = g["date"].str[:7]
        base = (
            g.sort_values("cloud_cover")
            .groupby("ym")
            .head(MAX_BASELINE_PER_MONTH_YEAR)
            .sort_values("date")
        )
        # --- monitor: every usable acquisition
        m = search_scenes(aoi, *cfg["monitor"], max_cloud=MAX_CLOUD_MONITOR)
        m = dedupe_by_date(m).sort_values("date")

        print(f"{site}: {len(base)} baseline + {len(m)} monitor scenes")
        for role, frame in [("baseline", base), ("monitor", m)]:
            for _, row in frame.iterrows():
                out = f"{out_dir}/{role}_{row['date']}.npz"
                if os.path.exists(out):
                    continue
                try:
                    bands, scl = fetch_scene(row["item"], crs, transform, h, w)
                except Exception as e:
                    print(f"  skip {row['date']}: {type(e).__name__} {e}")
                    continue
                np.savez_compressed(
                    out, bands=bands, scl=scl,
                    cloud=row["cloud_cover"], date=row["date"],
                )
                print(f"  {role} {row['date']} cloud={row['cloud_cover']:.1f}%")
