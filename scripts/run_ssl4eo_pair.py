"""Pairwise SSL4EO deep-feature change map between the two dates of each site.

Same math as the TESSERA map (cosine distance between per-pixel descriptors),
but the descriptor is the dense SSL4EO ResNet18 feature of a single scene, so
it works for any pair of dates. Exports GeoTIFFs (+ .qml styles) to
outputs/qgis/ and a comparison PNG per site.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

# torchgeo must be imported before rasterio/matplotlib (GDAL symbol clash
# segfaults the interpreter the other way round)
from run_semantic import build_encoder, deep_features

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from change_detection import normalize_to_reference, threshold_change, valid_mask
from export_qgis import qml_cosdist, QML_BINARY

OUT = "outputs/qgis"
TITLES = {
    "chiba": "Shimizu–Tomisato (Chiba)",
    "bassin": "SNCF Bassin",
}


def stretch_rgb(img):
    rgb = np.stack([img[2], img[1], img[0]]).astype(np.float64)
    out = np.zeros_like(rgb)
    for i in range(3):
        v = rgb[i][rgb[i] > 0]
        lo, hi = np.percentile(v, [2, 98])
        out[i] = np.clip((rgb[i] - lo) / max(hi - lo, 1), 0, 1)
    return np.moveaxis(out, 0, 2)


if __name__ == "__main__":
    model = build_encoder()
    for site in sys.argv[1:] or ["chiba", "bassin"]:
        before = sorted(glob.glob(f"data/{site}_before_*.tif"))[0]
        after = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
        d1 = os.path.basename(before).split("_")[-1][:-4]
        d2 = os.path.basename(after).split("_")[-1][:-4]
        with rasterio.open(before) as src:
            img1 = src.read()
            profile = src.profile
        with rasterio.open(after) as src:
            img2 = src.read()

        mask = valid_mask(img1, img2)
        img1n = normalize_to_reference(img1, img2.astype(np.float64), mask)

        f1 = deep_features(model, img1n)
        f2 = deep_features(model, img2.astype(np.float64))
        n1 = np.maximum(np.linalg.norm(f1, axis=0), 1e-8)
        n2 = np.maximum(np.linalg.norm(f2, axis=0), 1e-8)
        cos_dist = np.where(mask, 1.0 - (f1 * f2).sum(axis=0) / (n1 * n2), 0.0)

        # robust z-score threshold (same rule as the monitoring tiers); Otsu
        # over-flags on the heavy-tailed deep-feature distance distribution
        from skimage.morphology import remove_small_objects, binary_opening, disk

        med = np.median(cos_dist[mask])
        mad = np.median(np.abs(cos_dist[mask] - med)) * 1.4826
        z = (cos_dist - med) / max(mad, 1e-9)
        binary = remove_small_objects(
            binary_opening((z > 4.0) & mask, disk(1)), 4
        )
        print(f"{site} {d1}->{d2}: cos_dist max={cos_dist.max():.3f} "
              f"z4-threshold={med + 4 * mad:.3f} changed={binary.mean()*100:.2f}%")

        p = dict(profile)
        p.update(count=1, dtype="float32", compress="deflate")
        cd_path = f"{OUT}/{site}_SSL4EO_cosine_distance_{d1}_{d2}.tif"
        with rasterio.open(cd_path, "w", **p) as dst:
            dst.write(cos_dist.astype("float32"), 1)
        vmax = float(np.percentile(cos_dist[mask], 99.5))
        with open(cd_path.replace(".tif", ".qml"), "w") as f:
            f.write(qml_cosdist(vmax))

        p.update(dtype="uint8")
        bin_path = f"{OUT}/{site}_SSL4EO_change_binary_{d1}_{d2}.tif"
        with rasterio.open(bin_path, "w", **p) as dst:
            dst.write(binary.astype("uint8"), 1)
        with open(bin_path.replace(".tif", ".qml"), "w") as f:
            f.write(QML_BINARY)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=150)
        fig.suptitle(f"{TITLES[site]} — SSL4EO deep-feature pairwise change, "
                     f"{d1} → {d2}", fontsize=13)
        axes[0].imshow(stretch_rgb(img1)); axes[0].set_title(f"Before — {d1}", fontsize=10)
        axes[1].imshow(stretch_rgb(img2)); axes[1].set_title(f"After — {d2}", fontsize=10)
        im = axes[2].imshow(np.where(mask, cos_dist, np.nan), cmap="magma", vmax=vmax)
        axes[2].set_title("SSL4EO cosine distance", fontsize=10)
        fig.colorbar(im, ax=axes[2], fraction=0.045).ax.tick_params(labelsize=8)
        axes[3].imshow(stretch_rgb(img2))
        ov = np.zeros((*binary.shape, 4)); ov[binary] = [1.0, 0.17, 0.33, 0.75]
        axes[3].imshow(ov)
        axes[3].set_title("Change (robust z>4) on after-image", fontsize=10)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        png = f"outputs/{site}_ssl4eo_pair.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        print("wrote", cd_path, bin_path, png)
