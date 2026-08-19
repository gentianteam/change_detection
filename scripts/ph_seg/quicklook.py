import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from rasterio.plot import plotting_extent

for name in (sys.argv[1:] or SITES):
    src, img, gdf = load_site(name)
    step = max(1, max(img.shape[1:]) // 3000)
    rgb = stretch(rgb_of(img)[::step, ::step])
    fig, ax = plt.subplots(figsize=(18, 18 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb, extent=plotting_extent(src))
    cats = gdf["habcodes"].astype("category")
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats.cat.categories):
        gdf[cats == cat].boundary.plot(ax=ax, color=cmap(i), linewidth=1.5, label=cat)
    ax.legend(); ax.set_title(name); ax.set_axis_off()
    fig.savefig(OUT / "quicklooks" / f"{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(name, rgb.shape)
