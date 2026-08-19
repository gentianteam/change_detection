"""Post-process refined polygons per site:
  * ensemble: pixel majority vote of several methods -> 'vote' layer
  * overlap resolution between neighbouring polygons (pixel goes to the polygon whose ORIGINAL
    contained it, else to the nearest original)
  * geometry simplification (Douglas-Peucker, tol in m)
Writes <tag>/<site>_<method>_final.geojson
"""
import argparse, numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full")
ap.add_argument("--vote", default="grabcut2,sam2,sam2auto")
ap.add_argument("--methods", default="grabcut2,sam2,sam2auto,vote")
ap.add_argument("--res", type=float, default=0.5)
ap.add_argument("--simplify", type=float, default=0.6)
a = ap.parse_args()
outdir = OUT / a.tag
vote_methods = a.vote.split(",")

for site in SITES:
    if not (outdir / f"{site}_orig.geojson").exists():
        continue
    layers = {}
    for m in set(vote_methods) | set(a.methods.split(",")) | {"orig"}:
        f = outdir / f"{site}_{m}.geojson"
        if f.exists():
            layers[m] = gpd.read_file(f).set_index("pid")
    orig = layers["orig"]
    minx, miny, maxx, maxy = orig.total_bounds
    W = int(np.ceil((maxx - minx) / a.res)) + 2; H = int(np.ceil((maxy - miny) / a.res)) + 2
    tr = from_origin(minx - a.res, maxy + a.res, a.res, a.res)
    pids = list(orig.index)
    if H * W > 4e8:
        print(site, "too big for raster post-processing at this res"); continue

    def label_raster(gdf):
        return rasterize(((geom, i + 1) for i, geom in enumerate(gdf.geometry) if geom is not None and not geom.is_empty),
                         out_shape=(H, W), transform=tr, fill=0, dtype="int32")

    # per-pid boolean stacks are big; use label rasters and count votes per pixel per pid via per-polygon loop
    orig_lab = label_raster(orig.loc[pids])
    # nearest-original assignment for pixels not in any original
    dist, (iy, ix) = ndi.distance_transform_edt(orig_lab == 0, return_indices=True)
    nearest = orig_lab[iy, ix]

    out_layers = {}
    for m in a.methods.split(","):
        if m == "vote":
            stacks = [layers[v] for v in vote_methods if v in layers]
            if not stacks:
                continue
            votes = np.zeros((H, W), np.int16); claim = np.zeros((H, W), np.int32)
            # majority per polygon: pixel is in pid if >= half of the methods put it there
            per_pid_masks = {}
            for k, pid in enumerate(pids):
                cnt = np.zeros((H, W), np.int8)
                for L in stacks:
                    if pid in L.index and L.loc[pid].geometry is not None:
                        cnt += rasterize([(L.loc[pid].geometry, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="int8")
                per_pid_masks[pid] = cnt * 2 > len(stacks)
        else:
            if m not in layers:
                continue
            per_pid_masks = {}
            for pid in pids:
                g = layers[m].loc[pid].geometry if pid in layers[m].index else None
                per_pid_masks[pid] = rasterize([(g, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="int8").astype(bool) if g is not None and not g.is_empty else np.zeros((H, W), bool)
        # overlap resolution
        claim_count = np.zeros((H, W), np.int16)
        for pid in pids:
            claim_count += per_pid_masks[pid]
        overlap = claim_count > 1
        n_over = int(overlap.sum())
        recs = []
        for k, pid in enumerate(pids):
            msk = per_pid_masks[pid]
            if n_over:
                keep = ~overlap | (orig_lab == k + 1) | ((orig_lab == 0) & (nearest == k + 1))
                msk = msk & keep
            g = mask_to_polys(msk, tr, min_area_px=int(20 / a.res ** 2))
            if g is not None and a.simplify:
                g = g.simplify(a.simplify, preserve_topology=True).buffer(0)
            rec = {c: orig.loc[pid][c] for c in orig.columns if c not in ("geometry", "method")}
            rec.update(pid=pid, method=m, geometry=g)
            recs.append(rec)
        gdf = gpd.GeoDataFrame(recs, crs=orig.crs)
        save_gdf(gdf, outdir / f"{site}_{m}_final.geojson")
        print(f"{site:15s} {m:10s} overlap px resolved: {n_over:6d}  area orig {orig.area.sum()/1e4:.1f} ha -> {gdf.area.sum()/1e4:.1f} ha")
