# change_detection

Training-free remote-sensing experiments.

- `scripts/` — Sentinel-2 change detection (CVA, IR-MAD, SSL4EO/Tessera embeddings), report at `outputs/report.html`.
- `scripts/ph_seg/` — **Priority-habitat polygon border refinement** against VHR imagery with GrabCut and SAM 2.1
  (see [scripts/ph_seg/README.md](scripts/ph_seg/README.md)). Inputs: `ph_improved_seg/*.geojson` (+ the tifs, not versioned).
  Deliverables: `outputs/ph_seg/refined_polygons/`, report `outputs/ph_seg/report/report.html`.

Large rasters (`*.tif`, `data/`, `global_0.1_degree_*`) and intermediate outputs are git-ignored.
