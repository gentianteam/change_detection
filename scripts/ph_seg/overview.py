"""Site overview: original (red) vs a refined layer (yellow) over the whole image."""
import argparse, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, geopandas as gpd
from rasterio.plot import plotting_extent
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full"); ap.add_argument("--method", default="vote_final")
ap.add_argument("--sites", default=",".join(SITES)); ap.add_argument("--width", type=float, default=16)
a = ap.parse_args()
outdir = OUT / a.tag; figdir = outdir / "overview"; figdir.mkdir(exist_ok=True)
for site in a.sites.split(","):
    f = outdir / f"{site}_{a.method}.geojson"
    if not f.exists():
        print("missing", f); continue
    src, img, gdf = load_site(site)
    ref = gpd.read_file(f)
    step = max(1, max(img.shape[1:]) // 3000)
    rgb = stretch(rgb_of(img)[::step, ::step])
    fig, ax = plt.subplots(figsize=(a.width, a.width * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb, extent=plotting_extent(src))
    gdf.boundary.plot(ax=ax, color="red", linewidth=1.0, label="PHI original")
    ref.boundary.plot(ax=ax, color="yellow", linewidth=1.0, label=f"refined ({a.method})")
    b = gdf.total_bounds; ib = src.bounds
    ax.set_xlim(max(b[0], ib.left) - 20, min(b[2], ib.right) + 20); ax.set_ylim(max(b[1], ib.bottom) - 20, min(b[3], ib.top) + 20)
    ax.legend(loc="lower right"); ax.set_axis_off(); ax.set_title(site)
    fig.savefig(figdir / f"{site}_{a.method}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", figdir / f"{site}_{a.method}.png")
