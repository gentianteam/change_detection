"""Tier-1 alerting backtest: per-scene CVA anomaly vs monthly baseline composite,
with a consecutive-observation persistence rule.

For every monitor scene (chronological):
  1. valid = SCL in {4 veg, 5 bare, 6 water} for both scene and composite
  2. scene is radiometrically normalized to its month's baseline composite
  3. CVA magnitude -> robust z-score (median/MAD over valid pixels)
  4. counter: +1 while z > Z_ANOM and valid, reset to 0 on a valid calm
     observation, frozen while masked
  5. pixel alerts the first time counter >= PERSISTENCE

Outputs per site: outputs/monitor/<site>_tier1.npz with per-pixel first-alert
date index and per-scene stats.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import numpy as np

from change_detection import normalize_to_reference, cva_magnitude

GOOD_SCL = (4, 5, 6)
Z_ANOM = 4.0
PERSISTENCE = 3
MIN_VALID_FRAC = 0.30  # skip scenes where the AOI is mostly cloud

OUT = "outputs/monitor"


def scl_valid(scl):
    return np.isin(scl, GOOD_SCL)


def load_series(site, role):
    files = sorted(glob.glob(f"data/monitor/{site}/{role}_*.npz"))
    for f in files:
        d = np.load(f)
        yield str(d["date"]), d["bands"].astype(np.float64), d["scl"]


def monthly_composites(site):
    """Median composite + majority SCL validity per calendar month, pooling
    month +/-1 across all baseline years."""
    scenes = list(load_series(site, "baseline"))
    comps, comp_valid = {}, {}
    for month in range(1, 13):
        wanted = {(month - 2) % 12 + 1, month, month % 12 + 1}
        stack, vstack = [], []
        for date, bands, scl in scenes:
            if int(date[5:7]) in wanted:
                v = scl_valid(scl) & (bands.sum(axis=0) > 0)
                b = bands.copy()
                b[:, ~v] = np.nan
                stack.append(b)
                vstack.append(v)
        arr = np.stack(stack)  # (n, 4, h, w)
        comps[month] = np.nanmedian(arr, axis=0)
        comp_valid[month] = np.stack(vstack).sum(axis=0) >= 2
    return comps, comp_valid


def run_site(site):
    comps, comp_valid = monthly_composites(site)
    monitor = list(load_series(site, "monitor"))
    h, w = monitor[0][2].shape

    counter = np.zeros((h, w), dtype=np.int16)
    first_alert = np.full((h, w), -1, dtype=np.int16)  # scene index of alert
    stats = []
    kept_dates = []

    for date, bands, scl in monitor:
        month = int(date[5:7])
        comp = comps[month]
        cv = comp_valid[month] & ~np.isnan(comp.sum(axis=0))
        valid = scl_valid(scl) & (bands.sum(axis=0) > 0) & cv
        if valid.mean() < MIN_VALID_FRAC:
            continue

        ref = np.nan_to_num(comp)
        norm = normalize_to_reference(bands, ref, valid)
        mag = cva_magnitude(ref, norm)
        med = np.median(mag[valid])
        mad = np.median(np.abs(mag[valid] - med)) * 1.4826
        z = (mag - med) / max(mad, 1e-6)
        anom = (z > Z_ANOM) & valid

        idx = len(kept_dates)
        counter[anom] += 1
        counter[valid & ~anom] = 0
        newly = (counter >= PERSISTENCE) & (first_alert < 0)
        first_alert[newly] = idx

        kept_dates.append(date)
        stats.append(
            dict(date=date, valid_frac=float(valid.mean()),
                 anom_frac=float(anom[valid].mean()),
                 new_alerts=int(newly.sum()),
                 total_alerted=int((first_alert >= 0).sum()))
        )
        print(f"  {date} valid={valid.mean():.2f} anom={anom[valid].mean()*100:5.2f}% "
              f"new={newly.sum():4d} total={(first_alert>=0).sum()}")

    np.savez_compressed(
        f"{OUT}/{site}_tier1.npz",
        first_alert=first_alert,
        dates=np.array(kept_dates),
        valid_frac=np.array([s["valid_frac"] for s in stats]),
        anom_frac=np.array([s["anom_frac"] for s in stats]),
        new_alerts=np.array([s["new_alerts"] for s in stats]),
        total_alerted=np.array([s["total_alerted"] for s in stats]),
    )
    return stats


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sites = sys.argv[1:] or ["chiba", "bassin"]
    for site in sites:
        print(f"== {site}")
        run_site(site)
