"""Render per-site comparison figures: before/after RGB, method heatmaps,
binary change overlays, and a cross-method agreement map."""

import glob
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

OUT = "outputs"

AOI_FILES = {
    "chiba": "Site1_Tomisato_Chiba_AOI.geojson",
    "bassin": "Bassin_AOI.geojson",
}
TITLES = {
    "chiba": "Shimizu–Tomisato (Chiba, JP)",
    "bassin": "SNCF Bassin (FR)",
}


def stretch_rgb(img):
    """(4,h,w) uint16 BGRN -> display RGB with 2–98% stretch."""
    rgb = np.stack([img[2], img[1], img[0]]).astype(np.float64)
    out = np.zeros_like(rgb)
    for i in range(3):
        v = rgb[i][rgb[i] > 0]
        lo, hi = np.percentile(v, [2, 98])
        out[i] = np.clip((rgb[i] - lo) / max(hi - lo, 1), 0, 1)
    return np.moveaxis(out, 0, 2)


def aoi_pixel_outline(site):
    path = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
    with rasterio.open(path) as src:
        tfm, crs = src.transform, src.crs
    g = gpd.read_file(AOI_FILES[site]).to_crs(crs)
    outlines = []
    geom = g.geometry.unary_union
    polys = getattr(geom, "geoms", [geom])
    for poly in polys:
        xs, ys = poly.exterior.xy
        cols, rows = (~tfm) * (np.array(xs), np.array(ys))
        outlines.append((cols, rows))
    return outlines


def draw_aoi(ax, outlines):
    for cols, rows in outlines:
        ax.plot(cols, rows, color="#00E5FF", lw=1.2, alpha=0.9)


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    for site in ["chiba", "bassin"]:
        r = np.load(f"{OUT}/{site}_results.npz", allow_pickle=True)
        t = np.load(f"{OUT}/{site}_tessera.npz", allow_pickle=True)
        d1, d2 = r["dates"]
        y1, y2 = t["years"]
        rgb1, rgb2 = stretch_rgb(r["img1"]), stretch_rgb(r["img2"])
        outlines = aoi_pixel_outline(site)
        mask = r["mask"]

        agreement = (
            r["cva_bin"].astype(int)
            + r["irmad_bin"].astype(int)
            + t["tess_bin"].astype(int)
        )
        consensus = agreement >= 2

        fig, axes = plt.subplots(2, 4, figsize=(16, 8.6), dpi=150)
        fig.suptitle(
            f"{TITLES[site]} — Sentinel-2 change detection (no training)  |  "
            f"imagery {d1} → {d2},  TESSERA embeddings {y1} → {y2}",
            fontsize=13,
        )

        ax = axes[0, 0]
        ax.imshow(rgb1)
        draw_aoi(ax, outlines)
        style(ax, f"Before — {d1}")

        ax = axes[0, 1]
        ax.imshow(rgb2)
        draw_aoi(ax, outlines)
        style(ax, f"After — {d2}")

        ax = axes[0, 2]
        ax.imshow(rgb2)
        overlay = np.zeros((*consensus.shape, 4))
        overlay[consensus] = [1.0, 0.17, 0.33, 0.75]
        ax.imshow(overlay)
        draw_aoi(ax, outlines)
        style(ax, "Consensus change (≥2 of 3 methods) on after-image")
        ax.legend(
            handles=[Patch(facecolor="#FF2B55", alpha=0.75, label="changed")],
            loc="lower right", fontsize=8, framealpha=0.9,
        )

        ax = axes[0, 3]
        agree_cmap = ListedColormap(["#f2f2f0", "#fee0d2", "#fb6a4a", "#a50f15"])
        im = ax.imshow(agreement, cmap=agree_cmap, vmin=-0.5, vmax=3.5)
        draw_aoi(ax, outlines)
        style(ax, "Method agreement (0–3 methods flag change)")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, ticks=[0, 1, 2, 3])
        cb.ax.tick_params(labelsize=8)

        dndvi = np.where(mask, r["dndvi"], np.nan)
        lim = np.nanpercentile(np.abs(dndvi), 99)
        ax = axes[1, 0]
        im = ax.imshow(dndvi, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
        draw_aoi(ax, outlines)
        style(ax, "ΔNDVI (red = vegetation loss)")
        fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=8)

        ax = axes[1, 1]
        cva = np.where(mask, r["cva"], np.nan)
        im = ax.imshow(cva, cmap="magma",
                       vmax=np.nanpercentile(cva, 99.5))
        draw_aoi(ax, outlines)
        style(ax, "CVA magnitude (spectral change strength)")
        fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=8)

        ax = axes[1, 2]
        irm = np.where(mask, np.sqrt(r["irmad_stat"]), np.nan)
        im = ax.imshow(irm, cmap="magma", vmax=np.nanpercentile(irm, 99.5))
        draw_aoi(ax, outlines)
        style(ax, "IR-MAD √χ² (statistical change)")
        fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=8)

        ax = axes[1, 3]
        cd = np.where(t["mask"], t["cos_dist"], np.nan)
        im = ax.imshow(cd, cmap="magma", vmax=np.nanpercentile(cd, 99.5))
        draw_aoi(ax, outlines)
        style(ax, f"TESSERA embedding cosine distance {y1}→{y2}")
        fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = f"{OUT}/{site}_change_detection.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out_png,
              f"| consensus change: {consensus[mask].mean()*100:.2f}% of valid px")
