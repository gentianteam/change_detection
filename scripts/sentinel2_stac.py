"""Sentinel-2 L2A imagery access via the free Earth Search STAC API (AWS open data).

Mirrors the gentianteam GeoData-APIs/Sentinel2 workflow, but needs no API keys:
  1. `search_scenes(aoi, start, end, max_cloud)` -> GeoDataFrame of available scenes
  2. `download_rgbi(item, aoi, out_path)` -> clipped 4-band (B,G,R,NIR) GeoTIFF

Bands are read directly from the COGs on AWS (only the AOI window is fetched).
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.windows import from_bounds
from pystac_client import Client
from shapely.geometry import shape

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# Earth Search v1 asset keys for the 10 m RGBI bands
RGBI_ASSETS = ["blue", "green", "red", "nir"]


def search_scenes(aoi_gdf, start, end, max_cloud=20):
    """Return a GeoDataFrame of available Sentinel-2 L2A scenes intersecting the AOI.

    aoi_gdf : GeoDataFrame (any CRS)
    start, end : 'YYYY-MM-DD' strings
    max_cloud : max scene-level cloud cover percent
    """
    aoi_ll = aoi_gdf.to_crs(4326)
    geom = aoi_ll.geometry.unary_union.envelope.__geo_interface__
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        intersects=geom,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=500,
    )
    items = list(search.items())
    rows = []
    for it in items:
        rows.append(
            {
                "id": it.id,
                "datetime": it.datetime.strftime("%Y-%m-%d %H:%M"),
                "cloud_cover": it.properties.get("eo:cloud_cover"),
                "nodata_pct": it.properties.get("s2:nodata_pixel_percentage"),
                "mgrs_tile": it.properties.get("grid:code"),
                "item": it,
            }
        )
    gdf = gpd.GeoDataFrame(
        rows, geometry=[shape(it.geometry) for it in items], crs=4326
    )
    if len(gdf):
        gdf = gdf.sort_values("datetime").reset_index(drop=True)
    return gdf


def download_rgbi(item, aoi_gdf, out_path, buffer_m=0):
    """Download the 4 RGBI 10 m bands of a STAC item, clipped to the AOI bounds.

    Writes a 4-band uint16 GeoTIFF (B02 blue, B03 green, B04 red, B08 nir)
    in the scene's native UTM CRS. Returns the output path.
    """
    first_href = item.assets[RGBI_ASSETS[0]].href
    with rasterio.open(first_href) as src0:
        dst_crs = src0.crs
        aoi_utm = aoi_gdf.to_crs(dst_crs)
        minx, miny, maxx, maxy = aoi_utm.total_bounds
        if buffer_m:
            minx -= buffer_m
            miny -= buffer_m
            maxx += buffer_m
            maxy += buffer_m
        # snap to the 10 m grid so both dates align pixel-perfectly
        minx, miny = np.floor(minx / 10) * 10, np.floor(miny / 10) * 10
        maxx, maxy = np.ceil(maxx / 10) * 10, np.ceil(maxy / 10) * 10
        window = from_bounds(minx, miny, maxx, maxy, src0.transform)
        window = window.round_offsets().round_lengths()
        transform = src0.window_transform(window)
        h, w = int(window.height), int(window.width)

    bands = np.zeros((4, h, w), dtype=np.uint16)
    for i, key in enumerate(RGBI_ASSETS):
        href = item.assets[key].href
        with rasterio.open(href) as src:
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            win = win.round_offsets().round_lengths()
            bands[i] = src.read(1, window=win, boundless=True, fill_value=0)

    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=4,
        dtype="uint16",
        crs=dst_crs,
        transform=transform,
        compress="deflate",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands)
        dst.descriptions = ("blue", "green", "red", "nir")
    return out_path
