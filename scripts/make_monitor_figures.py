"""Figures for the monitoring backtest: tier-1 (spectral CVA) vs tier-2
(SSL4EO deep features), per site. Also exports first-alert GeoTIFFs for QGIS."""

import glob
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from datetime import datetime

OUT = "outputs/monitor"
TITLES = {
    "chiba": "Shimizu–Tomisato (Chiba) — monitor 2025, baseline 2022–2024",
    "bassin": "SNCF Bassin — monitor 2025-08→2026-07, baseline 2022–2025",
}
TIER_NAMES = {
    "tier1": "Tier 1 — spectral (CVA vs monthly composite)",
    "tier2": "Tier 2 — semantic (SSL4EO deep features)",
}


def after_gray(site):
    path = sorted(glob.glob(f"data/{site}_after_*.tif"))[0]
    with rasterio.open(path) as src:
        img = src.read().astype(np.float64)
        profile = src.profile
    g = 0.299 * img[2] + 0.587 * img[1] + 0.114 * img[0]
    v = g[g > 0]
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((g - lo) / (hi - lo), 0, 1), profile


def dts(dates):
    return [datetime.strptime(d, "%Y-%m-%d") for d in dates]


if __name__ == "__main__":
    for site in sys.argv[1:] or ["chiba", "bassin"]:
        gray, profile = after_gray(site)
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), dpi=150,
                                 gridspec_kw={"width_ratios": [1.15, 1, 1]})
        fig.suptitle(f"{TITLES[site]} — alerting backtest "
                     f"(z>4, {3} consecutive observations)", fontsize=13)

        for row, tier in enumerate(["tier1", "tier2"]):
            d = np.load(f"{OUT}/{site}_{tier}.npz")
            fa = d["first_alert"]
            dates = [str(x) for x in d["dates"]]
            x = dts(dates)

            ax = axes[row, 0]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            alert = np.ma.masked_where(fa < 0, fa)
            im = ax.imshow(alert, cmap="viridis", vmin=0, vmax=max(len(dates) - 1, 1))
            ax.set_title(f"{TIER_NAMES[tier]}\nalerts colored by first-alert date",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.045)
            ticks = np.linspace(0, len(dates) - 1, min(5, len(dates))).astype(int)
            cb.set_ticks(ticks)
            cb.set_ticklabels([dates[t] for t in ticks])
            cb.ax.tick_params(labelsize=7)

            ax = axes[row, 1]
            ax.step(x, d["total_alerted"] / 100.0, where="post",
                    color="#0E7C7B", lw=2)
            ax.bar(x, d["new_alerts"] / 100.0, width=4, color="#FF2B55",
                   alpha=0.8, label="new alerts")
            ax.set_title("Alerted area (hectares)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.25)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=8)

            ax = axes[row, 2]
            ax.plot(x, d["anom_frac"] * 100, "o-", color="#0E7C7B", ms=3, lw=1)
            ax.set_title("Anomalous fraction per usable scene (%) — "
                         "before persistence\n(marker spacing = cadence)",
                         fontsize=10)
            ax.grid(alpha=0.25)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=8)

            # QGIS export: day-of-monitoring-period of first alert (0 = no alert)
            p = dict(profile)
            p.update(count=1, dtype="int16", compress="deflate", nodata=-1)
            doy = np.full(fa.shape, -1, dtype=np.int16)
            t0 = dts([dates[0]])[0]
            for i, dt in enumerate(x):
                doy[fa == i] = (dt - t0).days
            with rasterio.open(f"{OUT}/{site}_{tier}_first_alert_day.tif",
                               "w", **p) as dst:
                dst.write(doy, 1)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = f"{OUT}/{site}_alerting.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
