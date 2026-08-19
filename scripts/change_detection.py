"""Training-free change detection methods for co-registered multispectral pairs.

Implemented methods (all unsupervised, no training data needed):
  - Relative radiometric normalization (robust per-band linear fit)
  - NDVI difference
  - Change Vector Analysis (CVA) magnitude
  - IR-MAD (Iteratively Reweighted Multivariate Alteration Detection,
    Nielsen 2007) with chi-square change probability
Thresholding via Otsu on the change magnitude.
"""

import numpy as np
from scipy.stats import chi2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk


def valid_mask(img1, img2):
    """Pixels valid (non-zero) in both images. img: (bands, h, w)."""
    return (img1.sum(axis=0) > 0) & (img2.sum(axis=0) > 0)


def normalize_to_reference(img, ref, mask):
    """Robust per-band linear normalization of `img` to match `ref`.

    Fits a line through the (img, ref) scatter using pixels in the central
    quantile range (pseudo-invariant majority), so illumination/atmosphere
    differences are removed while real changes remain.
    """
    out = img.astype(np.float64).copy()
    for b in range(img.shape[0]):
        x = img[b][mask].astype(np.float64)
        y = ref[b][mask].astype(np.float64)
        lo_x, hi_x = np.quantile(x, [0.02, 0.98])
        sel = (x > lo_x) & (x < hi_x)
        # two-pass robust fit: fit, drop large residuals, refit
        a, c = np.polyfit(x[sel], y[sel], 1)
        resid = np.abs(y[sel] - (a * x[sel] + c))
        keep = resid < 2 * resid.std()
        a, c = np.polyfit(x[sel][keep], y[sel][keep], 1)
        out[b] = a * img[b] + c
    return out


def ndvi(img):
    """img bands ordered (blue, green, red, nir)."""
    red = img[2].astype(np.float64)
    nir = img[3].astype(np.float64)
    return (nir - red) / np.maximum(nir + red, 1e-6)


def cva_magnitude(img1, img2):
    """Change vector magnitude across all bands (images should be normalized)."""
    d = img2.astype(np.float64) - img1.astype(np.float64)
    return np.sqrt((d**2).sum(axis=0))


def irmad(img1, img2, mask, max_iter=30, tol=1e-5):
    """Iteratively Reweighted MAD (Nielsen 2007).

    Returns (mad_variates (bands,h,w), chi2_stat (h,w), change_prob (h,w)).
    Change probability is P(chi2 > stat) complement: high value = change.
    """
    nb, h, w = img1.shape
    x = img1.reshape(nb, -1).astype(np.float64)[:, mask.ravel()]
    y = img2.reshape(nb, -1).astype(np.float64)[:, mask.ravel()]
    n = x.shape[1]
    wts = np.ones(n)

    for it in range(max_iter):
        sw = wts.sum()
        mx = (x * wts).sum(1, keepdims=True) / sw
        my = (y * wts).sum(1, keepdims=True) / sw
        xc, yc = x - mx, y - my
        sxx = (xc * wts) @ xc.T / sw
        syy = (yc * wts) @ yc.T / sw
        sxy = (xc * wts) @ yc.T / sw

        # canonical correlation analysis via generalized eigenproblem
        isxx = np.linalg.inv(sxx)
        isyy = np.linalg.inv(syy)
        evx, vx = np.linalg.eig(isxx @ sxy @ isyy @ sxy.T)
        order = np.argsort(-evx.real)
        rho2 = evx.real[order]
        vx = vx.real[:, order]
        # scale x-projections to unit variance
        vx = vx / np.sqrt(np.diag(vx.T @ sxx @ vx))
        # corresponding y-projections
        vy = isyy @ sxy.T @ vx
        vy = vy / np.sqrt(np.diag(vy.T @ syy @ vy))
        # sign convention: positive correlation
        sgn = np.sign(np.diag(vx.T @ sxy @ vy))
        vy = vy * sgn

        u = vx.T @ xc
        v = vy.T @ yc
        mads = u - v  # MAD variates, ordered by decreasing correlation
        var_mad = 2 * (1 - np.sqrt(np.clip(rho2, 0, 1)))
        var_mad = np.maximum(var_mad, 1e-12)
        stat = ((mads**2) / var_mad[:, None]).sum(0)
        new_wts = chi2.sf(stat, df=nb)  # no-change probability as weight

        if it > 0 and np.abs(new_wts - wts).mean() < tol:
            wts = new_wts
            break
        wts = new_wts

    mad_full = np.zeros((nb, h * w))
    stat_full = np.zeros(h * w)
    mad_full[:, mask.ravel()] = mads
    stat_full[mask.ravel()] = stat
    change_prob = 1 - chi2.sf(stat_full, df=nb)  # high = change
    return (
        mad_full.reshape(nb, h, w),
        stat_full.reshape(h, w),
        change_prob.reshape(h, w),
    )


def threshold_change(magnitude, mask, min_size=4):
    """Otsu threshold on the magnitude, cleaned with morphology."""
    vals = magnitude[mask]
    t = threshold_otsu(vals)
    binary = (magnitude > t) & mask
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=min_size)
    return binary, t
