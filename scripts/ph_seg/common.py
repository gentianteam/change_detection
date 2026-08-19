"""Shared helpers for the priority-habitat polygon refinement experiments."""
import numpy as np, rasterio, geopandas as gpd
from pathlib import Path
from rasterio.features import rasterize, shapes as rio_shapes
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ph_improved_seg"
OUT = ROOT / "outputs" / "ph_seg"

SITES = {
    "dunthropfarm":   ("DunthropFarm1_WVLegion_2025-08-17.tif", "phi_dunthropfarm.geojson"),
    "hrwallingford":  ("HRWallingford-HowberyPark_WV03_2026-07-07.tif", "phi_hrwallingford.geojson"),
    "essex":          ("EssexWildlifeTrust-GunnersPark_2025-06-30_VS30.tif", "phi_essex.geojson"),
    "blenheim":       ("BlenheimEstate11_WVLegion_2025-08-17.tif", "phi_nec-blenheimestate.geojson"),
    "northfieldfarm": ("NorthfieldFarm_WVLegion_2025-03-05.tif", "phi_northfieldfarm.geojson"),
    "millbrook":      ("MottMcDonald-East-West-Railway-Millbrook_PleiadesNeo_2025-05-20.tif", "phi_mm-millbrook.geojson"),
}


def load_site(name):
    tif, gj = SITES[name]
    src = rasterio.open(DATA / tif)
    img = src.read()  # (b,h,w) uint8
    gdf = gpd.read_file(DATA / gj).to_crs(src.crs)
    return src, img, gdf


def rgb_of(img):
    """uint8 (3,h,w) -> (h,w,3). Bands are R,G,B[,NIR]."""
    return np.moveaxis(img[:3], 0, -1)


def stretch(rgb, p=(2, 98)):
    out = np.empty_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        b = rgb[..., i].astype(np.float32)
        lo, hi = np.percentile(b[b > 0], p) if (b > 0).any() else (0, 255)
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1), 0, 1)
    return out


def rasterize_gdf(gdf, src_like, values=None, all_touched=False):
    if values is None:
        values = range(1, len(gdf) + 1)
    return rasterize(zip(gdf.geometry, values), out_shape=(src_like.height, src_like.width),
                     transform=src_like.transform, fill=0, all_touched=all_touched, dtype="int32")


def mask_to_polys(mask, transform, min_area_px=0):
    """Binary mask -> shapely MultiPolygon in raster CRS."""
    geoms = [shape(g) for g, v in rio_shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform) if v == 1]
    if min_area_px:
        pxa = abs(transform.a * transform.e)
        geoms = [g for g in geoms if g.area >= min_area_px * pxa]
    if not geoms:
        return None
    u = unary_union(geoms)
    return u


def iou(a, b):
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    return a.intersection(b).area / a.union(b).area


def window_for(geom, src, pad_m=30):
    """Pixel window (r0,r1,c0,c1) around geom bounds with padding, clipped."""
    minx, miny, maxx, maxy = geom.bounds
    minx -= pad_m; miny -= pad_m; maxx += pad_m; maxy += pad_m
    r0, c0 = src.index(minx, maxy)
    r1, c1 = src.index(maxx, miny)
    r0 = max(r0, 0); c0 = max(c0, 0); r1 = min(r1 + 1, src.height); c1 = min(c1 + 1, src.width)
    return r0, r1, c0, c1


def save_gdf(gdf, path):
    """to_file that survives pandas' StringDtype columns."""
    g = gdf.copy()
    for col in g.columns:
        if col != "geometry" and str(g[col].dtype) not in ("float64", "int64", "bool"):
            g[col] = g[col].astype(object).where(g[col].notna(), None)
    g.to_file(path, driver="GeoJSON")
