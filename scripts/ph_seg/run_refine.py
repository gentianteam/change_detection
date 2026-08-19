"""Run all refinement methods over all sites/polygons; write per-method GeoJSON + metrics CSV.

usage: python run_refine.py [--sites a,b] [--methods m1,m2] [--d 15] [--tag name]
"""
import argparse, time, json, numpy as np, pandas as pd, geopandas as gpd, cv2
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import sobel
from shapely.geometry import shape, box
from shapely.ops import unary_union
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.transform import Affine
from common import *
import refine

MAX_DIM = 2048  # working window max dimension in px


def valid_mask_full(img):
    return ~((img == 255).all(0) | (img == 0).all(0))


def footprint_polygon(valid, transform):
    v = ndi.binary_erosion(valid, iterations=2)
    return mask_to_polys(v, transform)


def prep_polygon(src, img, validfull, geom, d_m, pad_m):
    """Extract a working window around geom. Returns ctx dict or None."""
    r0, r1, c0, c1 = window_for(geom, src, pad_m)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return None
    win = img[:, r0:r1, c0:c1]
    valid = validfull[r0:r1, c0:c1]
    tr = src.transform * Affine.translation(c0, r0)
    scale = max(1, int(np.ceil(max(win.shape[1:]) / MAX_DIM)))
    if scale > 1:
        h, w = win.shape[1] // scale, win.shape[2] // scale
        win = np.stack([cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in win])
        valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        tr = tr * Affine.scale(scale)
    res = abs(tr.a)
    orig = rasterize([(geom, 1)], out_shape=valid.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    if orig.sum() < 30:
        return None
    d_px = max(2, int(round(d_m / res)))
    se = disk(d_px)
    core = ndi.binary_erosion(orig, se)
    if core.sum() < 10:  # very thin polygon: keep a skeleton-ish core
        dt = ndi.distance_transform_edt(orig)
        core = dt >= max(1, dt.max() * 0.5)
    outer = ~ndi.binary_dilation(orig, se)
    feats = refine.make_feats(win, valid)
    rgb = np.ascontiguousarray(np.moveaxis(win[:3], 0, -1))
    return dict(img=win, rgb=rgb, feats=feats, valid=valid, orig=orig, core=core, outer=outer,
                transform=tr, res=res, d_px=d_px, d_m=d_m)


def metrics(mask, ctx, orig_geom):
    """Proxy quality metrics for a mask (no ground truth available)."""
    feats, valid, res = ctx["feats"], ctx["valid"], ctx["res"]
    m = mask & valid
    if m.sum() == 0:
        return dict(area_ratio=0, iou_orig=0, edge_pct=np.nan, contrast=np.nan, in_std=np.nan, compact=np.nan)
    er = ndi.binary_erosion(m); bd = m & ~er  # boundary pixels
    # exclude boundary that lies on the image footprint edge
    bd &= ndi.binary_erosion(valid, iterations=3)
    # gradient magnitude on smoothed multiband
    grad = np.zeros(m.shape, np.float32)
    for c in range(min(feats.shape[-1], 5)):
        grad += sobel(ndi.gaussian_filter(feats[..., c], 1.0)) ** 2
    grad = np.sqrt(grad)
    ref = grad[valid]
    edge_pct = float(np.mean(np.searchsorted(np.sort(ref), grad[bd]) / len(ref))) if bd.any() else np.nan
    # ring contrast: inner ring vs outer ring (3 m) feature difference / pooled std, averaged over channels
    k = max(1, int(3 / res))
    inner = m & ~ndi.binary_erosion(m, disk(k)) & valid
    outerr = ~m & ndi.binary_dilation(m, disk(k)) & valid & ndi.binary_erosion(valid, iterations=3)
    if inner.sum() > 10 and outerr.sum() > 10:
        fi, fo = feats[inner][:, :5], feats[outerr][:, :5]
        contrast = float(np.mean(np.abs(fi.mean(0) - fo.mean(0)) / (0.5 * (fi.std(0) + fo.std(0)) + 1e-3)))
    else:
        contrast = np.nan
    gi = feats[..., 4] if feats.shape[-1] > 5 else feats[..., 3]
    in_std = float(gi[m].std())
    per = bd.sum() * res; area = m.sum() * res * res
    compact = float(per ** 2 / (4 * np.pi * area))
    orig = ctx["orig"] & valid
    inter = (m & orig).sum(); uni = (m | orig).sum()
    return dict(area_ratio=float(m.sum() / max(orig.sum(), 1)), iou_orig=float(inter / max(uni, 1)),
                edge_pct=edge_pct, contrast=contrast, in_std=in_std, compact=compact)


def mask_to_geom(mask, ctx):
    return mask_to_polys(mask, ctx["transform"], min_area_px=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--methods", default=",".join(refine.METHODS))
    ap.add_argument("--d", type=float, default=15.0, help="tolerance band half-width (m)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_polys", type=int, default=0)
    ap.add_argument("--adaptive", action="store_true", help="shrink tolerance for thin polygons: d_eff = clip(0.4*width, 4, d)")
    a = ap.parse_args()
    methods = a.methods.split(",")
    if any(m.startswith("sam2") for m in methods):
        import refine_sam
        refine.METHODS.update(refine_sam.METHODS)
    outdir = OUT / a.tag; outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in a.sites.split(","):
        src, img, gdf = load_site(site)
        validfull = valid_mask_full(img)
        fp = footprint_polygon(validfull, src.transform)
        results = {m: [] for m in methods}
        for i, row in gdf.iterrows():
            if a.max_polys and i >= a.max_polys:
                break
            geom = row.geometry
            geom_in = geom.intersection(fp) if fp is not None else geom
            if geom_in.is_empty or geom_in.area < 50:
                print(f"{site} #{i}: outside imagery, skipped"); continue
            outside_part = geom.difference(fp) if fp is not None else None
            d_eff = a.d
            if a.adaptive:
                width = 2 * geom_in.area / max(geom_in.length, 1)  # mean width proxy
                d_eff = float(np.clip(0.4 * width, 4, a.d))
            ctx = prep_polygon(src, img, validfull, geom_in, d_eff, pad_m=a.d + 20)
            if ctx is None:
                continue
            for m in methods:
                t = time.time()
                try:
                    mask = refine.METHODS[m](ctx)
                    if m != "orig":
                        mask = refine.clean(mask, ctx["core"], ctx["outer"], ctx["valid"])
                except Exception as e:
                    print(f"  {m} failed on {site}#{i}: {e}")
                    mask = ctx["orig"].copy()
                dt = time.time() - t
                met = metrics(mask, ctx, geom_in)
                g = mask_to_geom(mask, ctx)
                if g is not None and outside_part is not None and not outside_part.is_empty:
                    g = unary_union([g, outside_part])
                rec = {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for k, v in row.drop("geometry").items()}
                rec.update(dict(pid=i, site=site, method=m, geometry=g))
                results[m].append(rec)
                rows.append(dict(site=site, pid=i, hab=row["habcodes"], method=m, secs=round(dt, 2), d_eff=ctx["d_m"],
                                 area_ha=geom_in.area / 1e4, win_px=ctx["orig"].size, res=ctx["res"], **met))
                print(f"{site} #{i:3d} {row['habcodes'][:12]:12s} {m:12s} {dt:6.1f}s  IoU={met['iou_orig']:.2f} "
                      f"edge={met['edge_pct']:.2f} contrast={met['contrast']:.2f} area×={met['area_ratio']:.2f}", flush=True)
        for m in methods:
            if results[m]:
                save_gdf(gpd.GeoDataFrame(results[m], crs=src.crs), outdir / f"{site}_{m}.geojson")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"metrics_{a.sites.replace(',','-')}_{a.methods.split(',')[-1]}.csv", index=False)
    print(df.groupby("method")[["secs", "iou_orig", "edge_pct", "contrast", "in_std", "compact"]].mean().round(3))


if __name__ == "__main__":
    main()
