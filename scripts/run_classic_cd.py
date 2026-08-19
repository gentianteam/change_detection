"""Run all classic training-free change detection methods on both site pairs.

Saves per-site results to outputs/<site>_results.npz and GeoTIFF change maps.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import numpy as np
import rasterio

from change_detection import (
    cva_magnitude,
    irmad,
    ndvi,
    normalize_to_reference,
    threshold_change,
    valid_mask,
)

OUT = "outputs"


def load_pair(site):
    before = sorted(glob.glob(f"data/{site}_before_*.tif"))[0]
    after = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
    with rasterio.open(before) as src:
        img1 = src.read()
        profile = src.profile
    with rasterio.open(after) as src:
        img2 = src.read()
    dates = (
        os.path.basename(before).split("_")[-1][:-4],
        os.path.basename(after).split("_")[-1][:-4],
    )
    return img1, img2, profile, dates


def save_geotiff(path, arr, profile, dtype="float32"):
    p = dict(profile)
    p.update(count=1, dtype=dtype, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for site in ["chiba", "bassin"]:
        img1, img2, profile, dates = load_pair(site)
        mask = valid_mask(img1, img2)
        print(f"== {site}: {dates[0]} -> {dates[1]}, shape {img1.shape}, "
              f"valid {mask.mean()*100:.1f}%")

        # radiometric normalization: match 'before' to 'after'
        img1n = normalize_to_reference(img1, img2, mask)
        img2f = img2.astype(np.float64)

        # 1. NDVI difference
        dndvi = ndvi(img2f) - ndvi(img1n)

        # 2. CVA magnitude
        cva = cva_magnitude(img1n, img2f)
        cva_bin, cva_t = threshold_change(cva, mask)

        # 3. IR-MAD
        mads, stat, prob = irmad(img1n, img2f, mask)
        # Otsu on sqrt of the chi-square statistic (chi2 tails are too permissive
        # when the scene contains large genuine change)
        irmad_bin, irmad_t = threshold_change(np.sqrt(stat), mask)

        print(f"   CVA change: {cva_bin.mean()*100:.2f}% of pixels (t={cva_t:.0f})")
        print(f"   IR-MAD change (Otsu): {irmad_bin.mean()*100:.2f}% of pixels")
        print(f"   NDVI diff range: {dndvi[mask].min():.2f}..{dndvi[mask].max():.2f}")

        np.savez_compressed(
            f"{OUT}/{site}_results.npz",
            img1=img1, img2=img2, img1n=img1n, mask=mask, dates=np.array(dates),
            dndvi=dndvi, cva=cva, cva_bin=cva_bin,
            irmad_stat=stat, irmad_prob=prob, irmad_bin=irmad_bin,
        )
        save_geotiff(f"{OUT}/{site}_cva_magnitude.tif", cva, profile)
        save_geotiff(f"{OUT}/{site}_irmad_prob.tif", prob, profile)
        save_geotiff(f"{OUT}/{site}_change_binary_irmad.tif", irmad_bin, profile, "uint8")
