# Refining priority-habitat polygon borders with imagery (no training)

Snaps existing habitat polygons (PHI GeoJSON) to the edges visible in VHR imagery, using
**GrabCut** (CPU) and/or **SAM 2.1** (GPU). Every method only moves the border inside a tolerance
band (±15 m by default); the polygon core stays inside, far-outside stays outside.

```
scripts/ph_seg/
  run_refine.py    run one or more methods on one or more sites -> GeoJSON per site/method + metrics
  postprocess.py   majority vote of methods, overlap resolution between neighbours, simplification
  panels.py        before/after crops per polygon (QA figures)
  overview.py      whole-site before/after figure
  refine.py        classical methods (grabcut, grabcut2, watershed, ...) + band/clean-up
  refine_sam.py    SAM2 methods (sam2, sam2iter, sam2tile, sam2auto)
  common.py        site list, I/O helpers
```

Inputs live in `ph_improved_seg/`, outputs go to `outputs/ph_seg/<tag>/`.

## Setup

Python env: `~/miniconda3/envs/habitat-mapping/bin/python` (rasterio, geopandas, scikit-image,
opencv, scikit-learn, torch + `sam2` installed). SAM 2.1 weights (`facebook/sam2.1-hiera-large`,
~900 MB) are downloaded from HuggingFace on first use and cached; SAM needs a CUDA GPU (~2–3 GB).
GrabCut needs no GPU.

```bash
cd ~/change_detection/scripts/ph_seg
P=~/miniconda3/envs/habitat-mapping/bin/python
```

Sites are registered in `common.py` (`SITES` = name -> (tif, geojson)); add a line there for a new
site. Imagery must be uint8 with bands R,G,B[,NIR]; polygons any CRS (reprojected to the raster).

## 1. GrabCut (`grabcut2`) — CPU, ~2 s per polygon

```bash
# one site
$P run_refine.py --sites dunthropfarm --methods orig,grabcut2 --adaptive --tag mytest

# all sites, in parallel (one process per site)
for s in dunthropfarm hrwallingford essex blenheim northfieldfarm millbrook; do
  $P run_refine.py --sites $s --methods orig,grabcut2 --adaptive --tag mytest > /tmp/gc_$s.log 2>&1 &
done; wait
```

Outputs: `outputs/ph_seg/mytest/<site>_grabcut2.geojson` (+ `<site>_orig.geojson` = the input
clipped/re-indexed the same way, useful for comparison) and `metrics_*.csv`.

How it works: NIR–R–G false-colour image; polygon eroded by *d* = definite foreground, beyond the
polygon dilated by *d* = definite background, the band in between = "probably background"; 5
GrabCut iterations (GMM colour models + graph cut). `grabcut` (without the 2) learns the foreground
model from the whole original polygon instead of the eroded core — worse when the polygon
contains hedges/shadows.

## 2. SAM 2.1 (`sam2iter`) — GPU, ~2–3 s per polygon

```bash
# one site
$P run_refine.py --sites dunthropfarm --methods orig,sam2iter --adaptive --tag mytest

# all sites (keep ONE process for SAM so the model loads once and GPU memory stays ~3 GB)
$P run_refine.py --methods sam2iter --adaptive --tag mytest > /tmp/sam.log 2>&1 &
```

Outputs: `outputs/ph_seg/mytest/<site>_sam2iter.geojson`.

How it works: RGB window around the polygon -> SAM2 image encoder; prompts derived from the
polygon (bounding box, positive points deep inside the eroded core, negative points in a ring
outside, and the polygon itself as a low-res mask prompt); best-IoU candidate is fed back as mask
prompt for 2 more rounds. Variants: `sam2` (box+points only), `sam2tile` (same prompts on
native-resolution 640 px tiles, more conservative), `sam2auto` (automatic masks, not recommended).

Running several SAM processes at once works but each loads its own model (~2–3 GB GPU each).

## 3. Combine, tidy, look

```bash
# majority vote of the methods you ran + overlap resolution + 0.6 m simplification
$P postprocess.py --tag mytest --vote grabcut2,sam2iter --methods grabcut2,sam2iter,vote
#   -> outputs/ph_seg/mytest/<site>_{grabcut2,sam2iter,vote}_final.geojson

# QA figures
$P overview.py --tag mytest --method vote_final --sites dunthropfarm          # whole site
$P panels.py --tag mytest --sites dunthropfarm --pids 0,1 --size 5 \
             --methods orig,grabcut2_final,sam2iter_final,vote_final         # per polygon, red=orig, yellow=refined
#   -> outputs/ph_seg/mytest/{overview,panels}/*.png
```

With two voters the vote needs both to agree (ties count as outside); with three
(`grabcut2,sam2iter,sam2tile`) it is a real majority — that is the delivered configuration.

## Options worth knowing (`run_refine.py`)

| flag | default | meaning |
|---|---|---|
| `--sites a,b` | all six | site names from `common.SITES` |
| `--methods m1,m2` | all | any of `orig grabcut grabcut2 superpixel pixelclf watershed randomwalker gac sam2 sam2mask sam2iter sam2tile sam2auto`; always include `orig` once per tag (postprocess/panels need it) |
| `--d 15` | 15 | tolerance half-width in metres: how far a border may move |
| `--adaptive` | off | shrink *d* for thin polygons: `d_eff = clip(0.4 × mean width, 4, d)` — recommended |
| `--tag name` | run | output folder `outputs/ph_seg/<tag>/` |
| `--max_polys N` | 0 (all) | only the first N polygons of each site (quick tests) |

`postprocess.py`: `--vote m1,m2,...` (voters), `--methods ...` (which layers to write, `vote` included),
`--res 0.5` (vote raster, m), `--simplify 0.6` (Douglas-Peucker tolerance, m).

## Output format

GeoJSON in the raster CRS (EPSG:27700 here) with the original attributes plus `pid` (row index in the
input file) and `method`. Parts of a polygon outside the imagery footprint are kept unchanged.
Polygons entirely outside the imagery are skipped. A polygon can come back as a MultiPolygon if the
model drops a neck narrower than the tolerance band (see the report for details).

## Notes

- Metrics in `metrics_*.csv` are proxies (no ground truth): `contrast` (inner vs outer ring feature
  difference along the new border), `edge_pct` (image gradient under the border), `iou_orig`,
  `area_ratio`, `compact`. Use them for sanity, trust the panels.
- GrabCut is deterministic (OpenCV RNG seeded). SAM is deterministic for a given GPU/driver.
- Full report of the method comparison: `outputs/ph_seg/report/report.html`.
