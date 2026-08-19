"""Training-free refinement of priority-habitat polygon borders against VHR imagery.

Every method receives the same inputs for one polygon:
    feats  : (h,w,C) float32 features (R,G,B,NIR,NDVI...) at working resolution
    rgb    : (h,w,3) uint8 for SAM/GrabCut
    core   : bool mask, definitely inside  (orig eroded by d)
    outer  : bool mask, definitely outside (beyond orig dilated by d)
    orig   : bool mask of the original polygon
    valid  : bool mask of valid imagery
and returns a bool mask. The band (~core & ~outer & valid) is the only region a
method is allowed to change; we enforce that afterwards.
"""
import numpy as np, cv2, warnings
from scipy import ndimage as ndi
from skimage.segmentation import slic, random_walker, watershed, morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.filters import sobel
from skimage.color import rgb2lab
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_holes, remove_small_objects
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- features
def make_feats(img_win, valid):
    """img_win (b,h,w) uint8 with bands R,G,B[,NIR] -> (h,w,C) float32 in ~[0,1]."""
    x = img_win.astype(np.float32) / 255.0
    r, g, b = x[0], x[1], x[2]
    ch = [r, g, b]
    if x.shape[0] >= 4:
        n = x[3]
        ndvi = (n - r) / (n + r + 1e-3)
        ch += [n, ndvi]
    else:
        # pseudo greenness for RGB-only imagery
        exg = (2 * g - r - b)
        ch += [exg]
    f = np.stack(ch, -1)
    # local texture: std of the first (red) band, 5x5
    m = ndi.uniform_filter(r, 5); m2 = ndi.uniform_filter(r * r, 5)
    tex = np.sqrt(np.clip(m2 - m * m, 0, None))
    f = np.concatenate([f, tex[..., None]], -1)
    f[~valid] = 0
    return f


def clean(mask, core, outer, valid, min_frac=0.002, radius=2):
    """Enforce band constraint + light morphological regularisation."""
    m = mask.copy()
    m |= core
    m &= ~outer
    m &= valid
    if radius:
        m = binary_opening(m, disk(radius))
        m = binary_closing(m, disk(radius))
        m |= core
        m &= ~outer
    # drop small blobs / fill small holes relative to polygon size
    n = max(int(min_frac * core.sum()), 20)
    m = remove_small_objects(m, n)
    m = remove_small_holes(m, n)
    m |= core; m &= ~outer; m &= valid
    return m


# ----------------------------------------------------------------------------- methods
def m_orig(ctx):
    return ctx["orig"].copy()


def m_grabcut(ctx, iters=5, use_nir=True, core_only=False):
    rgb, core, outer, orig = ctx["rgb"], ctx["core"], ctx["outer"], ctx["orig"]
    if core_only:  # foreground model learnt from the core only; whole band starts as probable background
        orig = core
    img = rgb
    if use_nir and ctx["img"].shape[0] >= 4:
        img = np.moveaxis(ctx["img"][[3, 0, 1]], 0, -1).copy()  # false colour NIR,R,G
    gc = np.full(orig.shape, cv2.GC_PR_BGD, np.uint8)
    gc[orig] = cv2.GC_PR_FGD
    gc[core] = cv2.GC_FGD
    gc[outer] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)  # GrabCut's GMM init uses k-means++ with OpenCV's RNG -> make it reproducible
    try:
        cv2.grabCut(np.ascontiguousarray(img), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return orig.copy()
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def _sp_features(feats, labels):
    n = labels.max() + 1
    C = feats.shape[-1]
    out = np.zeros((n, C), np.float32)
    cnt = np.bincount(labels.ravel(), minlength=n).astype(np.float32)
    for c in range(C):
        out[:, c] = np.bincount(labels.ravel(), weights=feats[..., c].ravel(), minlength=n) / np.maximum(cnt, 1)
    return out, cnt


def m_superpixel(ctx, seg_m=4.0, k_neigh=15):
    """SLIC superpixels; band superpixels labelled by kNN on superpixel mean features
    trained on core (in) vs outer-ring (out) superpixels."""
    feats, core, outer, valid, res = ctx["feats"], ctx["core"], ctx["outer"], ctx["valid"], ctx["res"]
    band = ~core & ~outer & valid
    ring = outer & ndi.binary_dilation(band, disk(int(15 / res)))  # near-outside as negatives
    px_per_seg = (seg_m / res) ** 2
    zone = valid & ndi.binary_dilation(band, disk(int(20 / res)))  # only segment an annulus around the band
    n_seg = int(np.clip(zone.sum() / px_per_seg, 50, 4000))
    lab = rgb2lab(ctx["rgb"]).astype(np.float32)
    x = np.concatenate([lab / np.array([100, 128, 128], np.float32), feats[..., 3:]], -1)
    labels = slic(x, n_segments=n_seg, compactness=0.05, sigma=1, channel_axis=-1, mask=zone, start_label=1)
    spf, cnt = _sp_features(feats, labels)
    core_frac = np.bincount(labels.ravel(), weights=core.ravel(), minlength=len(cnt)) / np.maximum(cnt, 1)
    ring_frac = np.bincount(labels.ravel(), weights=ring.ravel(), minlength=len(cnt)) / np.maximum(cnt, 1)
    band_frac = np.bincount(labels.ravel(), weights=band.ravel(), minlength=len(cnt)) / np.maximum(cnt, 1)
    pos = np.where(core_frac > 0.8)[0]; neg = np.where(ring_frac > 0.8)[0]
    cand = np.where(band_frac > 0.2)[0]
    if len(pos) < 3 or len(neg) < 3 or len(cand) == 0:
        return ctx["orig"].copy()
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([spf[pos], spf[neg]]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = StandardScaler().fit(X)
    k = min(k_neigh, len(pos), len(neg))
    clf = KNeighborsClassifier(k, weights="distance").fit(sc.transform(X), y)
    p = clf.predict_proba(sc.transform(spf[cand]))[:, 1]
    lut = np.zeros(len(cnt), bool)
    lut[pos] = True
    lut[cand[p >= 0.5]] = True
    lut[cand[p < 0.5]] = False
    return lut[labels] & valid


def m_pixelclf(ctx, smooth_sigma=1.5):
    """Per-polygon pixel classifier (RandomForest) core-vs-ring on smoothed features, then threshold."""
    from sklearn.ensemble import RandomForestClassifier
    feats, core, outer, valid, res = ctx["feats"], ctx["core"], ctx["outer"], ctx["valid"], ctx["res"]
    band = ~core & ~outer & valid
    ring = outer & ndi.binary_dilation(band, disk(int(15 / res)))
    f = np.stack([ndi.gaussian_filter(feats[..., c], smooth_sigma) for c in range(feats.shape[-1])], -1)
    f = np.concatenate([f, feats], -1)
    rng = np.random.default_rng(0)
    pi = np.flatnonzero(core.ravel()); ni = np.flatnonzero(ring.ravel())
    if len(pi) < 50 or len(ni) < 50:
        return ctx["orig"].copy()
    pi = rng.choice(pi, min(4000, len(pi)), replace=False); ni = rng.choice(ni, min(4000, len(ni)), replace=False)
    X = f.reshape(-1, f.shape[-1])
    clf = RandomForestClassifier(100, min_samples_leaf=5, n_jobs=8, random_state=0).fit(
        np.concatenate([X[pi], X[ni]]), np.r_[np.ones(len(pi)), np.zeros(len(ni))])
    bi = np.flatnonzero(band.ravel())
    p = np.zeros(band.size, np.float32)
    p[bi] = clf.predict_proba(X[bi])[:, 1]
    p = p.reshape(band.shape)
    p = ndi.gaussian_filter(p, 1.0)
    return core | (band & (p >= 0.5))


def m_watershed(ctx):
    """Marker-based watershed on multi-band gradient (edges) inside the band."""
    feats, core, outer, valid = ctx["feats"], ctx["core"], ctx["outer"], ctx["valid"]
    grad = np.zeros(core.shape, np.float32)
    for c in range(min(feats.shape[-1], 5)):
        grad += sobel(ndi.gaussian_filter(feats[..., c], 1.0)) ** 2
    grad = np.sqrt(grad)
    markers = np.zeros(core.shape, np.int32)
    markers[core] = 1; markers[outer | ~valid] = 2
    ws = watershed(grad, markers)
    return (ws == 1) & valid


def m_randomwalker(ctx, beta=130):
    feats, core, outer, valid = ctx["feats"], ctx["core"], ctx["outer"], ctx["valid"]
    labels = np.zeros(core.shape, np.int32)
    labels[core] = 1; labels[outer | ~valid] = 2
    # downsample if big for speed
    h, w = core.shape
    scale = 1
    while (h / scale) * (w / scale) > 1.5e6:
        scale += 1
    data = feats[..., :5]
    if scale > 1:
        data_s = data[::scale, ::scale]; labels_s = labels[::scale, ::scale]
    else:
        data_s, labels_s = data, labels
    try:
        rw = random_walker(data_s, labels_s, beta=beta, mode="cg_j", channel_axis=-1, tol=1e-3)
    except Exception:
        rw = random_walker(data_s, labels_s, beta=beta, mode="bf", channel_axis=-1)
    m = rw == 1
    if scale > 1:
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m & valid


def m_gac(ctx, iterations=None, balloon=0):
    """Morphological geodesic active contour on the band, from the original polygon."""
    feats, core, outer, valid, res = ctx["feats"], ctx["core"], ctx["outer"], ctx["valid"], ctx["res"]
    # edge map from luminance + greenness (NDVI or ExG)
    lum = feats[..., :3].mean(-1)
    x = 0.5 * lum + 0.5 * (feats[..., 4] if feats.shape[-1] > 5 else feats[..., 3])
    gimg = inverse_gaussian_gradient(x, alpha=200, sigma=1.5)
    it = iterations or int(1.5 * ctx["d_px"])
    ls = morphological_geodesic_active_contour(gimg, it, init_level_set=ctx["orig"].astype(np.int8),
                                               smoothing=2, balloon=balloon, threshold="auto")
    return ls.astype(bool) & valid


METHODS = {
    "orig": m_orig,
    "grabcut": m_grabcut,
    "grabcut2": lambda ctx: m_grabcut(ctx, core_only=True),
    "superpixel": m_superpixel,
    "pixelclf": m_pixelclf,
    "watershed": m_watershed,
    "randomwalker": m_randomwalker,
    "gac": m_gac,
}
