"""SAM2 prompted refinement. Prompts are derived from the existing polygon:
   box (bbox of polygon), positive points inside the core, negative points in the outer ring,
   optionally the polygon itself as a low-res mask prompt, iterated.
"""
import numpy as np, cv2, torch
from scipy import ndimage as ndi
from skimage.morphology import disk
from sklearn.cluster import KMeans

_PRED = None


def predictor():
    global _PRED
    if _PRED is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _PRED = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")
    return _PRED


def _sample_points(mask, k, seed=0):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 2))
    if len(ys) <= k:
        return np.stack([xs, ys], 1).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), min(len(ys), 5000), replace=False)
    pts = np.stack([xs[idx], ys[idx]], 1).astype(float)
    km = KMeans(k, n_init=3, random_state=seed).fit(pts)
    out = []
    for c in km.cluster_centers_:
        j = np.argmin(((pts - c) ** 2).sum(1))
        out.append(pts[j])
    return np.array(out)


def _prompts(ctx, n_pos=None, n_neg=None):
    orig, core, outer, valid, res = ctx["orig"], ctx["core"], ctx["outer"], ctx["valid"], ctx["res"]
    area_ha = orig.sum() * res * res / 1e4
    if n_pos is None:
        n_pos = int(np.clip(2 + area_ha, 3, 12))
    if n_neg is None:
        n_neg = int(np.clip(2 + area_ha, 3, 12))
    # positive points: deep inside core (top 50% of distance transform)
    dt = ndi.distance_transform_edt(core)
    deep = dt >= max(1, 0.4 * dt.max())
    pos = _sample_points(deep, n_pos)
    # negative points: ring outside, d..d+12m from polygon
    ring = outer & ndi.binary_dilation(~outer, disk(int(12 / res))) & valid
    ring &= ndi.binary_erosion(valid, iterations=3)
    neg = _sample_points(ring, n_neg, seed=1)
    ys, xs = np.nonzero(orig)
    pad = int(3 / res)
    box = np.array([max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, orig.shape[1] - 1), min(ys.max() + pad, orig.shape[0] - 1)], float)
    return pos, neg, box


def _run(ctx, use_mask=False, iters=1, use_box=True, multimask=True):
    p = predictor()
    rgb = ctx["rgb"]
    orig = ctx["orig"]
    pos, neg, box = _prompts(ctx)
    pts = np.concatenate([pos, neg]) if len(neg) else pos
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        p.set_image(rgb)
        mask_in = None
        if use_mask:
            lo = cv2.resize(orig.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
            mask_in = ((lo - 0.5) * 20.0)[None]  # logits: +10 inside, -10 outside
        best = None
        for it in range(iters):
            masks, scores, logits = p.predict(point_coords=pts, point_labels=lab, box=box if use_box else None,
                                              mask_input=mask_in, multimask_output=multimask)
            # choose the candidate with best IoU vs original polygon
            ious = [((m.astype(bool) & orig).sum() / max((m.astype(bool) | orig).sum(), 1)) for m in masks]
            j = int(np.argmax(ious))
            best = masks[j].astype(bool)
            mask_in = logits[j][None]
            multimask = False
    return best


def m_sam2(ctx):
    return _run(ctx, use_mask=False, iters=1)


def m_sam2mask(ctx):
    return _run(ctx, use_mask=True, iters=1)


def m_sam2iter(ctx):
    return _run(ctx, use_mask=True, iters=3)


METHODS = {"sam2": m_sam2, "sam2mask": m_sam2mask, "sam2iter": m_sam2iter}


# ---------------------------------------------------------------- SAM segments as superpixels
_GEN = None


def generator():
    global _GEN
    if _GEN is None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        _GEN = SAM2AutomaticMaskGenerator(predictor().model, points_per_side=24, points_per_batch=64,
                                          pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                          crop_n_layers=0, min_mask_region_area=50, multimask_output=True)
    return _GEN


def m_sam2auto(ctx, thr=0.5):
    """All SAM2 masks in the window act as superpixels; a segment is inside if >thr of it lies in the
    original polygon. Small segments override large ones (processed large->small)."""
    orig, core, outer, valid = ctx["orig"], ctx["core"], ctx["outer"], ctx["valid"]
    rgb = ctx["rgb"]
    h, w = orig.shape
    scale = 1.0
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        anns = generator().generate(rgb)
    anns = sorted(anns, key=lambda a: -a["area"])
    out = orig.copy().astype(np.int8)  # start from original; -1 unknown
    lab = np.zeros(orig.shape, np.int8)  # 0 untouched, 1 in, 2 out
    for a in anns:
        m = a["segmentation"]
        if scale != 1.0:
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m &= valid
        if m.sum() == 0:
            continue
        frac = (m & orig).sum() / m.sum()
        # only segments that touch the band matter; keep them if mostly inside
        lab[m] = 1 if frac >= thr else 2
    res = orig.copy()
    res[lab == 1] = True
    res[lab == 2] = False
    return res


METHODS["sam2auto"] = m_sam2auto


# ---------------------------------------------------------------- tiled SAM2 (native resolution along the border)
def m_sam2tile(ctx, tile=640, stride=448, n_pts=3, use_mask=True):
    """Run SAM2 on native-resolution tiles that intersect the tolerance band. In every tile the
    prompt is: positive points in core, negative points in the outer ring, plus the original polygon
    (clipped to the tile) as a low-res mask prompt. Tile logits are averaged."""
    p = predictor()
    orig, core, outer, valid, res = ctx["orig"], ctx["core"], ctx["outer"], ctx["valid"], ctx["res"]
    rgb = ctx["rgb"]
    H, W = orig.shape
    band = ~core & ~outer & valid
    ring = outer & ndi.binary_dilation(~outer, disk(int(12 / res))) & valid
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - tile, 0) + 1, stride)) + ([max(H - tile, 0)] if (H - tile) % stride else [])
    xs = list(range(0, max(W - tile, 0) + 1, stride)) + ([max(W - tile, 0)] if (W - tile) % stride else [])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                b = band[y0:y1, x0:x1]
                if b.sum() < 20:
                    continue
                c = core[y0:y1, x0:x1]; r = ring[y0:y1, x0:x1]; o = orig[y0:y1, x0:x1]
                pos = _sample_points(ndi.binary_erosion(c, disk(3)) if c.sum() > 500 else c, n_pts, seed=y0 + x0)
                neg = _sample_points(r, n_pts, seed=y0 + x0 + 1)
                if len(pos) == 0:
                    continue
                pts = np.concatenate([pos, neg]) if len(neg) else pos
                lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                p.set_image(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
                mask_in = None
                if use_mask:
                    lo = cv2.resize(o.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
                    mask_in = ((lo - 0.5) * 20.0)[None]
                masks, scores, logits = p.predict(point_coords=pts, point_labels=lab, mask_input=mask_in, multimask_output=True)
                ious = [((m.astype(bool) & o).sum() / max((m.astype(bool) | o).sum(), 1)) for m in masks]
                j = int(np.argmax(ious))
                lg = cv2.resize(logits[j].astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                # down-weight tile borders
                wy = np.minimum(np.arange(y1 - y0) + 1, np.arange(y1 - y0)[::-1] + 1).clip(max=64) / 64.0
                wx = np.minimum(np.arange(x1 - x0) + 1, np.arange(x1 - x0)[::-1] + 1).clip(max=64) / 64.0
                wgt = wy[:, None] * wx[None, :]
                acc[y0:y1, x0:x1] += np.clip(lg, -20, 20) * wgt; cnt[y0:y1, x0:x1] += wgt
    out = orig.copy()
    seen = cnt > 0
    out[seen] = (acc[seen] / cnt[seen]) > 0
    return out


METHODS["sam2tile"] = m_sam2tile
