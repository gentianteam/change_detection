"""Tier-2 semantic alerting: same monitoring loop as tier 1, but the per-pixel
descriptor is a dense deep feature from SSL4EO's Sentinel-2 pretrained ResNet18
(TorchGeo SENTINEL2_RGB_MOCO weights) instead of raw spectra.

Features: layer1 (stride 4) + layer2 (stride 8) maps, bilinearly upsampled to
full 10 m resolution, channel-wise L2-normalized and concatenated (192-d).
Change score per scene = cosine distance between the scene's and the monthly
composite's feature vectors. Robust z + persistence, as in tier 1.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import numpy as np
import torch
import torch.nn.functional as F
from torchgeo.models import resnet18, ResNet18_Weights

from run_alerting import (
    monthly_composites, load_series, scl_valid,
    MIN_VALID_FRAC, Z_ANOM, PERSISTENCE, OUT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_encoder():
    model = resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def deep_features(model, bands):
    """bands (4,h,w) float reflectance -> (192,h,w) L2-normalized features.

    Uses the RGB bands (indices 2,1,0) scaled to [0,1] as the RGB-MoCo
    weights expect natural-range inputs.
    """
    h, w = bands.shape[1:]
    rgb = np.stack([bands[2], bands[1], bands[0]]) / 10000.0
    x = torch.from_numpy(np.clip(rgb, 0, 1)).float()[None].to(DEVICE)
    # pad to multiple of 32 for clean strided convs
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")

    f = model.conv1(x); f = model.bn1(f); f = model.act1(f); f = model.maxpool(f)
    f1 = model.layer1(f)      # stride 4
    f2 = model.layer2(f1)     # stride 8
    feats = []
    for fm in (f1, f2):
        up = F.interpolate(fm, size=x.shape[-2:], mode="bilinear",
                           align_corners=False)[..., :h, :w]
        feats.append(F.normalize(up, dim=1))
    return torch.cat(feats, dim=1)[0].cpu().numpy()


def run_site(site, model):
    comps, comp_valid = monthly_composites(site)
    comp_feats = {
        m: deep_features(model, np.nan_to_num(comps[m])) for m in comps
    }
    monitor = list(load_series(site, "monitor"))
    h, w = monitor[0][2].shape

    counter = np.zeros((h, w), dtype=np.int16)
    first_alert = np.full((h, w), -1, dtype=np.int16)
    stats, kept_dates = [], []

    for date, bands, scl in monitor:
        month = int(date[5:7])
        cv = comp_valid[month] & ~np.isnan(comps[month].sum(axis=0))
        valid = scl_valid(scl) & (bands.sum(axis=0) > 0) & cv
        if valid.mean() < MIN_VALID_FRAC:
            continue

        feat = deep_features(model, bands)
        ref = comp_feats[month]
        # cosine distance between 192-d unit-normalized block features
        fa = feat / np.maximum(np.linalg.norm(feat, axis=0), 1e-8)
        fb = ref / np.maximum(np.linalg.norm(ref, axis=0), 1e-8)
        dist = 1.0 - (fa * fb).sum(axis=0)

        med = np.median(dist[valid])
        mad = np.median(np.abs(dist[valid] - med)) * 1.4826
        z = (dist - med) / max(mad, 1e-6)
        anom = (z > Z_ANOM) & valid

        idx = len(kept_dates)
        counter[anom] += 1
        counter[valid & ~anom] = 0
        newly = (counter >= PERSISTENCE) & (first_alert < 0)
        first_alert[newly] = idx

        kept_dates.append(date)
        stats.append((valid.mean(), anom[valid].mean(),
                      int(newly.sum()), int((first_alert >= 0).sum())))
        print(f"  {date} valid={valid.mean():.2f} anom={anom[valid].mean()*100:5.2f}% "
              f"new={newly.sum():4d} total={(first_alert>=0).sum()}")

    s = np.array(stats)
    np.savez_compressed(
        f"{OUT}/{site}_tier2.npz",
        first_alert=first_alert, dates=np.array(kept_dates),
        valid_frac=s[:, 0], anom_frac=s[:, 1],
        new_alerts=s[:, 2].astype(int), total_alerted=s[:, 3].astype(int),
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    model = build_encoder()
    sites = sys.argv[1:] or ["chiba", "bassin"]
    for site in sites:
        print(f"== {site} (device={DEVICE})")
        run_site(site, model)
