"""Visual comparison panels: one row per polygon, one column per method (crop around polygon)."""
import sys, argparse, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, geopandas as gpd, cv2
from rasterio.plot import plotting_extent
from rasterio.windows import Window
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dev"); ap.add_argument("--sites", default=None)
ap.add_argument("--methods", default=None); ap.add_argument("--max_polys", type=int, default=12)
ap.add_argument("--pids", default=None); ap.add_argument("--size", type=float, default=3.2); ap.add_argument("--zoom", default=None, help="xmin,ymin,xmax,ymax fraction of crop")
a = ap.parse_args()
outdir = OUT / a.tag; figdir = outdir / "panels"; figdir.mkdir(exist_ok=True)
sites = a.sites.split(",") if a.sites else [p.stem.split("_")[0] for p in sorted(outdir.glob("*_orig.geojson"))]
for site in sites:
    files = {p.stem[len(site) + 1:]: p for p in outdir.glob(f"{site}_*.geojson")}
    methods = a.methods.split(",") if a.methods else [m for m in ["orig", "grabcut", "superpixel", "pixelclf", "watershed", "randomwalker", "gac", "sam2", "sam2mask", "sam2iter", "combo"] if m in files]
    layers = {m: gpd.read_file(files[m]).set_index("pid") for m in set(methods) | {"orig"}}
    src = rasterio.open(DATA / SITES[site][0])
    pids = [int(p) for p in a.pids.split(",")] if a.pids else list(layers["orig"].index)[: a.max_polys]
    n = len(methods)
    fig, axes = plt.subplots(len(pids), n, figsize=(a.size * n, a.size * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        g0 = layers["orig"].loc[pid].geometry
        minx, miny, maxx, maxy = g0.bounds
        pad = 0.15 * max(maxx - minx, maxy - miny) + 10
        r0, c0 = src.index(minx - pad, maxy + pad); r1, c1 = src.index(maxx + pad, miny - pad)
        r0, c0 = max(r0, 0), max(c0, 0); r1, c1 = min(r1, src.height), min(c1, src.width)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        img = src.read([1, 2, 3], window=win)
        step = max(1, max(img.shape[1:]) // int(300 * a.size))
        rgb = stretch(np.moveaxis(img[:, ::step, ::step], 0, -1))
        ext = plotting_extent(src.read(1, window=win)[::step, ::step], src.window_transform(win) * rasterio.Affine.scale(step))
        for c, m in enumerate(methods):
            ax = axes[r, c]; ax.imshow(rgb, extent=ext)
            gpd.GeoSeries([g0], crs=src.crs).boundary.plot(ax=ax, color="red", linewidth=1.0, alpha=0.9)
            if m != "orig" and pid in layers[m].index:
                gm = layers[m].loc[pid].geometry
                if gm is not None:
                    gpd.GeoSeries([gm], crs=src.crs).boundary.plot(ax=ax, color="yellow", linewidth=1.0)
            if a.zoom:
                zx0, zy0, zx1, zy1 = [float(v) for v in a.zoom.split(",")]
                W, H = ext[1] - ext[0], ext[3] - ext[2]
                ax.set_xlim(ext[0] + zx0 * W, ext[0] + zx1 * W); ax.set_ylim(ext[2] + zy0 * H, ext[2] + zy1 * H)
            else:
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_axis_off()
            if r == 0: ax.set_title(m, fontsize=11)
            if c == 0: ax.text(0.01, 0.99, f"#{pid} {layers['orig'].loc[pid]['habcodes']}", transform=ax.transAxes, va="top", color="w", fontsize=8, bbox=dict(fc="k", alpha=0.5, lw=0))
    fig.tight_layout(); fn = figdir / (f"{site}.png" if not a.pids else f"{site}_p{a.pids.replace(',','-')}{'_z' if a.zoom else ''}_{'-'.join(methods)}.png"); fig.savefig(fn, dpi=100); plt.close(fig)
    print("wrote", fn)
