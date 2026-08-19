"""Change detection from TESSERA foundation-model embeddings (no training).

For each site, fetch the 128-dim annual embeddings for two years, align them
onto the same grid as the downloaded RGBI imagery, and compute per-pixel
cosine distance between years. High distance = the land surface's annual
spectral-temporal signature changed.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

from geotessera import GeoTessera
from change_detection import threshold_change

OUT = "outputs"

# earliest/latest years with published embeddings per site (checked via API)
SITE_YEARS = {"chiba": (2020, 2024), "bassin": (2017, 2025)}


def rgbi_grid(site):
    path = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
    with rasterio.open(path) as src:
        return src.crs, src.transform, src.height, src.width


def fetch_aligned_embedding(gt, site, year, bbox_ll, crs, transform, h, w):
    """Fetch embedding mosaic for bbox (lon/lat) and warp onto the RGBI grid."""
    arr, tfm, mosaic_crs = gt.fetch_mosaic_for_region(bbox_ll, year=year,
                                                      target_crs=str(crs))
    if arr.ndim == 3 and arr.shape[2] == 128:  # (h, w, c) -> (c, h, w)
        arr = np.moveaxis(arr, 2, 0)
    if not isinstance(tfm, Affine):
        tfm = Affine(*tfm[:6])
    out = np.zeros((arr.shape[0], h, w), dtype=np.float32)
    reproject(
        arr.astype(np.float32), out,
        src_transform=tfm, src_crs=mosaic_crs,
        dst_transform=transform, dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    return out


if __name__ == "__main__":
    import geopandas as gpd

    os.makedirs(OUT, exist_ok=True)
    gt = GeoTessera()
    for site, (y1, y2) in SITE_YEARS.items():
        crs, transform, h, w = rgbi_grid(site)
        # bbox of the RGBI grid in lon/lat, padded a little
        left, top = transform * (0, 0)
        right, bottom = transform * (w, h)
        import rasterio.warp as rw
        bbox_ll = rw.transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
        pad = 0.002
        bbox_ll = (bbox_ll[0] - pad, bbox_ll[1] - pad,
                   bbox_ll[2] + pad, bbox_ll[3] + pad)

        e1 = fetch_aligned_embedding(gt, site, y1, bbox_ll, crs, transform, h, w)
        e2 = fetch_aligned_embedding(gt, site, y2, bbox_ll, crs, transform, h, w)

        n1 = np.linalg.norm(e1, axis=0)
        n2 = np.linalg.norm(e2, axis=0)
        mask = (n1 > 1e-6) & (n2 > 1e-6)
        cos_sim = (e1 * e2).sum(axis=0) / np.maximum(n1 * n2, 1e-12)
        cos_dist = np.where(mask, 1.0 - cos_sim, 0.0)
        eucl = np.linalg.norm(e2 - e1, axis=0)

        tess_bin, tess_t = threshold_change(cos_dist, mask)
        print(f"{site}: TESSERA {y1}->{y2}  cos_dist range "
              f"{cos_dist[mask].min():.3f}..{cos_dist[mask].max():.3f}  "
              f"changed {tess_bin.mean()*100:.2f}% (t={tess_t:.3f})")

        np.savez_compressed(
            f"{OUT}/{site}_tessera.npz",
            cos_dist=cos_dist, eucl=eucl, mask=mask,
            tess_bin=tess_bin, years=np.array([y1, y2]),
        )
        p = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                 crs=crs, transform=transform, compress="deflate")
        with rasterio.open(f"{OUT}/{site}_tessera_cosdist.tif", "w", **p) as dst:
            dst.write(cos_dist.astype(np.float32), 1)
