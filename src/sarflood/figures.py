"""
Slide figures, built from the pipeline's existing inputs and outputs.

    python -m sarflood figures --config config/arakawa_2022.yaml

Reads data/raw/*.tif, outputs/flood_*.tif, outputs/metrics.csv and
outputs/recall_by_landcover_VH.csv (run `sarflood run` first) and writes PNGs
to outputs/figures/. Nothing here changes detection or evaluation.

Colours (Okabe-Ito, colour-blind safe) are shared by every figure so "hit",
"missed" and "false alarm" always look the same.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .acquire import sar_filename  # noqa: E402
from .basemap import get_basemap  # noqa: E402
from .export import polygonize  # noqa: E402
from .grid import Grid, read_to_grid  # noqa: E402
from .masks import remove_small_patches  # noqa: E402
from .pipeline import build_context, ensure_linear  # noqa: E402
from .preprocess import linear_to_db  # noqa: E402

# ---------------------------------------------------------------- styling
HIT = "#0072B2"          # detected and in GSI reference
MISSED = "#D55E00"       # in GSI reference, not detected
FALSE_ALARM = "#F0E442"  # detected, not in GSI reference
BOTH = "#332288"         # VV/VH comparison: detected by both (indigo, not green: basemap is green fields)
VH_ONLY = "#56B4E9"
VV_ONLY = "#CC79A7"
CROPLAND = "#E6C84F"
BUILT_UP = "#5A5A5A"
OTHER_LC = "#EFEFEF"
WATER_LC = "#9BC6E8"

FILL_ALPHA = 0.55
DPI = 200
MAP_WIDTH_IN = 9.0
LEGEND_WIDTH_IN = 3.6
FONT = {"title": 15, "legend": 12, "small": 9}

FLOOD_MARGIN_M = 1000                                   # flood maps: GSI extent + 1 km
LANDCOVER_ZOOM = (362_900, 4_218_800, 366_100, 4_221_600)  # Fujisawa / Shimokajiya + cropland
GSI_LABEL = "GSI reference (survey outline)"
EVAL_NOTE = "Scores computed within 300 m of the GSI reference area."


def _gsi_style() -> dict:
    """Black outline with a white halo: readable on SAR and on optical imagery."""
    return {
        "color": "black",
        "linewidth": 1.4,
        "path_effects": [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()],
    }


def _gsi_handle() -> Line2D:
    return Line2D([0], [0], label=GSI_LABEL, **_gsi_style())


# ---------------------------------------------------------------- geometry helpers
def _window(grid: Grid, bounds: tuple[float, float, float, float]) -> tuple[slice, slice, tuple]:
    """Row/column slices for map ``bounds`` (left, bottom, right, top), clipped to the grid."""
    res = abs(grid.transform.a)
    left0, top0 = grid.transform.c, grid.transform.f
    c0 = max(int(np.floor((bounds[0] - left0) / res)), 0)
    c1 = min(int(np.ceil((bounds[2] - left0) / res)), grid.width)
    r0 = max(int(np.floor((top0 - bounds[3]) / res)), 0)
    r1 = min(int(np.ceil((top0 - bounds[1]) / res)), grid.height)
    extent = (left0 + c0 * res, left0 + c1 * res, top0 - r1 * res, top0 - r0 * res)
    return slice(r0, r1), slice(c0, c1), extent


def _flood_bounds(reference: np.ndarray, grid: Grid, margin_m: float) -> tuple[float, float, float, float]:
    res = abs(grid.transform.a)
    rows, cols = np.nonzero(reference)
    left = grid.transform.c + cols.min() * res - margin_m
    right = grid.transform.c + (cols.max() + 1) * res + margin_m
    top = grid.transform.f - rows.min() * res + margin_m
    bottom = grid.transform.f - (rows.max() + 1) * res - margin_m
    return (left, bottom, right, top)


def _subgrid(grid: Grid, rows: slice, cols: slice) -> Grid:
    from affine import Affine

    res = abs(grid.transform.a)
    t = Affine(res, 0, grid.transform.c + cols.start * res, 0, -res, grid.transform.f - rows.start * res)
    return Grid(grid.crs, t, cols.stop - cols.start, rows.stop - rows.start)


def _overlay(layers: list[tuple[np.ndarray, str]], alpha: float = FILL_ALPHA) -> np.ndarray:
    """RGBA image from (mask, colour) pairs; later layers paint over earlier ones."""
    shape = layers[0][0].shape
    out = np.zeros((*shape, 4), dtype="float32")
    for mask, colour in layers:
        out[mask] = (*matplotlib.colors.to_rgb(colour), alpha)
    return out


# ---------------------------------------------------------------- map furniture
def _nice_length(span_m: float) -> float:
    """Largest round length (km) that is at most a quarter of the map width."""
    best = 0.25
    for km in (0.25, 0.5, 1, 2, 5, 10):
        if km * 1000 <= span_m / 4:
            best = km
    return best * 1000


def _scale_bar(ax, extent) -> None:
    left, right, bottom, top = extent
    length = _nice_length(right - left)
    x0 = right - (right - left) * 0.04 - length
    y0 = bottom + (top - bottom) * 0.05
    h = (top - bottom) * 0.012
    half = length / 2
    ax.add_patch(plt.Rectangle((x0, y0), half, h, facecolor="black", edgecolor="black", zorder=20))
    ax.add_patch(plt.Rectangle((x0 + half, y0), half, h, facecolor="white", edgecolor="black", zorder=20))
    label = f"{length / 1000:g} km"
    ax.text(x0 + length / 2, y0 + h * 1.8, label, ha="center", va="bottom", fontsize=FONT["legend"],
            zorder=20, path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def _north_arrow(ax, extent) -> None:
    left, right, bottom, top = extent
    x = right - (right - left) * 0.05
    y = top - (top - bottom) * 0.17
    dy = (top - bottom) * 0.09
    ax.annotate("N", xy=(x, y + dy), xytext=(x, y), ha="center", va="top", fontsize=FONT["legend"] + 1,
                fontweight="bold", zorder=20,
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2},
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def _finish_map(ax, extent) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    _scale_bar(ax, extent)
    _north_arrow(ax, extent)


def _map_figure(extent, extra_width_in: float = LEGEND_WIDTH_IN, ncols: int = 1):
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig = plt.figure(figsize=(MAP_WIDTH_IN + extra_width_in, MAP_WIDTH_IN * aspect + 1.1), dpi=DPI)
    return fig


def _credit(ax, text: str) -> None:
    """Small source line just below the map frame."""
    ax.text(0.0, -0.012, text, transform=ax.transAxes, fontsize=FONT["small"], color="#444444",
            ha="left", va="top", wrap=False)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def _date(text: str) -> str:
    """'2022-07-22' -> '22 Jul 2022' (left unchanged if it is not an ISO date)."""
    try:
        d = date.fromisoformat(str(text))
    except ValueError:
        return str(text)
    return f"{d.day} {d.strftime('%b %Y')}"


# ---------------------------------------------------------------- data loading
class _Inputs:
    """Everything the figures need, read once."""

    def __init__(self, cfg: dict, overwrite_basemap: bool = False):
        self.cfg = cfg
        raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["outputs"])
        ctx = build_context(cfg)
        self.grid, self.reference, self.region, self.landcover = ctx.grid, ctx.reference, ctx.region, ctx.landcover
        self.gsi = gpd.read_file(cfg["reference"]["gsi_flood"]).to_crs(self.grid.crs)

        self.detected = {}
        for pol in cfg["sar"]["polarizations"]:
            path = out / f"flood_{pol}.tif"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run `python -m sarflood run` first")
            self.detected[pol] = read_to_grid(path, self.grid, Resampling.nearest) > 0.5

        self.metrics = pd.read_csv(out / "metrics.csv", index_col=0)
        lc = pd.read_csv(out / "recall_by_landcover_VH.csv")
        self.recall_lc_vh = dict(zip(lc["class"], lc["recall"]))
        self.raw = raw
        self.out = out
        self._basemap = None
        self._overwrite_basemap = overwrite_basemap

    @property
    def basemap(self):
        if self._basemap is None:
            self._basemap = get_basemap(self.cfg, self.grid, self._overwrite_basemap)
        return self._basemap

    def vh_db(self, epoch: str) -> np.ndarray:
        arr = read_to_grid(self.raw / sar_filename(epoch, "VH"), self.grid)
        return linear_to_db(ensure_linear(arr))


def _categories(inp: _Inputs, pol: str) -> dict[str, np.ndarray]:
    d, ref = inp.detected[pol], inp.reference
    return {"hit": d & ref, "missed": ref & ~d, "false": d & ~ref}


def _hit_missed_false_handles(prefix: str = "") -> list:
    return [
        Patch(facecolor=HIT, alpha=0.8, label=f"{prefix}Hit: detected, in GSI reference"),
        Patch(facecolor=MISSED, alpha=0.8, label=f"{prefix}Missed: in GSI, not detected"),
        Patch(facecolor=FALSE_ALARM, alpha=0.8, label=f"{prefix}False alarm: detected, not in GSI"),
    ]


def _legend(ax, handles, title: str | None = None) -> None:
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                    fontsize=FONT["legend"], title=title, title_fontsize=FONT["legend"], alignment="left")
    leg._legend_box.align = "left"


# ---------------------------------------------------------------- figures
def sar_before_after(inp: _Inputs, out_dir: Path) -> list[Path]:
    """Before/after VH in dB, same grey stretch, GSI outline. SAR only."""
    before, after = inp.vh_db("before"), inp.vh_db("after")
    both = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    vmin, vmax = np.percentile(both, [2, 98])
    extent = (inp.grid.bounds[0], inp.grid.bounds[2], inp.grid.bounds[1], inp.grid.bounds[3])
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")

    paths = []
    for epoch, arr in (("before", before), ("after", after)):
        when = _date(inp.cfg["sar"][epoch][0])
        fig = _map_figure(extent, extra_width_in=1.4)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        im = ax.imshow(arr, cmap=gray, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        inp.gsi.boundary.plot(ax=ax, **_gsi_style())
        _finish_map(ax, extent)
        label = "before the flood" if epoch == "before" else "after the flood"
        ax.set_title(f"Sentinel-1 VH backscatter, {label} ({when})", fontsize=FONT["title"], loc="left")
        cax = ax.inset_axes([1.02, 0.25, 0.025, 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("VH backscatter (dB)", fontsize=FONT["legend"])
        cb.ax.tick_params(labelsize=FONT["small"] + 1)
        ax.legend(handles=[_gsi_handle()], loc="upper left", fontsize=FONT["legend"], framealpha=0.85)
        _credit(ax, "Contains modified Copernicus Sentinel data 2022 (Sentinel-1 GRD, descending orbit 46). "
                     f"Same grey stretch on both dates: {vmin:.1f} to {vmax:.1f} dB.")
        paths.append(_save(fig, out_dir / f"sar_{epoch}_VH.png"))
    return paths


def flood_map_slide(inp: _Inputs, out_dir: Path) -> Path:
    """VH detection vs GSI on the optical basemap, with headline scores."""
    rows, cols, extent = _window(inp.grid, _flood_bounds(inp.reference, inp.grid, FLOOD_MARGIN_M))
    cat = _categories(inp, "VH")
    m = inp.metrics.loc["VH"]

    fig = _map_figure(extent)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(inp.basemap.rgb[rows, cols], extent=extent, interpolation="bilinear")
    ax.imshow(_overlay([(cat["false"][rows, cols], FALSE_ALARM), (cat["missed"][rows, cols], MISSED),
                        (cat["hit"][rows, cols], HIT)]), extent=extent, interpolation="nearest")
    inp.gsi.boundary.plot(ax=ax, **_gsi_style())
    _finish_map(ax, extent)
    ax.set_title(f"Sentinel-1 VH flood detection vs GSI survey   |   IoU {m['iou']:.2f}   ·   "
                 f"Precision {m['precision']:.2f}   ·   Recall {m['recall']:.2f}",
                 fontsize=FONT["title"], loc="left")
    _legend(ax, _hit_missed_false_handles() + [_gsi_handle()])
    _credit(ax, f"{inp.basemap.credit}. Flood reference: GSI inundation estimate. {EVAL_NOTE}")
    return _save(fig, out_dir / "flood_map_VH_slide.png")


def vv_vs_vh(inp: _Inputs, out_dir: Path) -> Path:
    """Which polarization detects what, against the GSI reference."""
    rows, cols, extent = _window(inp.grid, _flood_bounds(inp.reference, inp.grid, FLOOD_MARGIN_M))
    vh, vv, ref = inp.detected["VH"], inp.detected["VV"], inp.reference
    layers = [
        ((ref & ~vh & ~vv)[rows, cols], MISSED),
        ((vh & ~vv)[rows, cols], VH_ONLY),
        ((vv & ~vh)[rows, cols], VV_ONLY),
        ((vh & vv)[rows, cols], BOTH),
    ]
    fig = _map_figure(extent)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(inp.basemap.rgb[rows, cols], extent=extent, interpolation="bilinear")
    ax.imshow(_overlay(layers), extent=extent, interpolation="nearest")
    inp.gsi.boundary.plot(ax=ax, **_gsi_style())
    _finish_map(ax, extent)

    mvh, mvv = inp.metrics.loc["VH"], inp.metrics.loc["VV"]
    ax.set_title("VH vs VV flood detection against the GSI survey", fontsize=FONT["title"], loc="left")
    handles = [
        Patch(facecolor=BOTH, alpha=0.8, label="Detected by both"),
        Patch(facecolor=VH_ONLY, alpha=0.8, label="VH only"),
        Patch(facecolor=VV_ONLY, alpha=0.8, label="VV only"),
        Patch(facecolor=MISSED, alpha=0.8, label="GSI flood missed by both"),
        _gsi_handle(),
        Patch(facecolor="none", edgecolor="none", label=""),
        Patch(facecolor="none", edgecolor="none",
              label=f"VH: IoU {mvh['iou']:.2f}, built-up recall {mvh['recall_built_up']:.1%}"),
        Patch(facecolor="none", edgecolor="none",
              label=f"VV: IoU {mvv['iou']:.2f}, built-up recall {mvv['recall_built_up']:.1%}"),
    ]
    _legend(ax, handles)
    _credit(ax, f"{inp.basemap.credit}. Flood reference: GSI inundation estimate. {EVAL_NOTE}")
    return _save(fig, out_dir / "vv_vs_vh_overlay.png")


def landcover_zoom(inp: _Inputs, out_dir: Path) -> Path:
    """Fujisawa / Shimokajiya: cropland floods detected, built-up floods missed."""
    rows, cols, extent = _window(inp.grid, LANDCOVER_ZOOM)
    if rows.stop - rows.start < 10 or cols.stop - cols.start < 10:   # zoom box not on this grid
        rows, cols, extent = _window(inp.grid, _flood_bounds(inp.reference, inp.grid, FLOOD_MARGIN_M))
    sub = _subgrid(inp.grid, rows, cols)
    cat = _categories(inp, "VH")
    lc = inp.landcover[rows, cols]

    from scipy import ndimage

    built = ndimage.binary_closing(lc == 50, structure=np.ones((3, 3)), iterations=2)
    built = remove_small_patches(built, min_pixels=150)
    built_outline = polygonize(built, sub, min_area_m2=20_000)
    built_outline["geometry"] = built_outline.buffer(10).buffer(-10).simplify(15)

    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    main_w, side_w = 7.6, 3.4
    fig = plt.figure(figsize=(main_w + side_w + 3.4, main_w * aspect + 1.3), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, main_w / (main_w + side_w + 3.4), 1.0])
    side = fig.add_axes([(main_w + 0.3) / (main_w + side_w + 3.4), 0.47,
                         side_w / (main_w + side_w + 3.4), 0.45])

    ax.imshow(inp.basemap.rgb[rows, cols], extent=extent, interpolation="bilinear")
    ax.imshow(_overlay([(cat["false"][rows, cols], FALSE_ALARM), (cat["missed"][rows, cols], MISSED),
                        (cat["hit"][rows, cols], HIT)]), extent=extent, interpolation="nearest")
    if len(built_outline):
        built_outline.boundary.plot(ax=ax, color="white", linewidth=1.3, linestyle=(0, (4, 2)),
                                    path_effects=[pe.Stroke(linewidth=2.6, foreground=BUILT_UP), pe.Normal()])
    inp.gsi.boundary.plot(ax=ax, **_gsi_style())
    _finish_map(ax, extent)

    classes = np.full(lc.shape, 0, dtype="uint8")          # 0 other
    classes[lc == 40] = 1                                  # cropland
    classes[lc == 50] = 2                                  # built-up
    classes[np.isin(lc, (80, 90))] = 3                     # water / wetland
    side.imshow(classes, cmap=ListedColormap([OTHER_LC, CROPLAND, BUILT_UP, WATER_LC]), vmin=0, vmax=3,
                extent=extent, interpolation="nearest")
    inp.gsi.boundary.plot(ax=side, color="black", linewidth=0.9)
    side.set_xlim(extent[0], extent[1])
    side.set_ylim(extent[2], extent[3])
    side.set_xticks([])
    side.set_yticks([])
    side.set_xlabel("")
    side.set_ylabel("")
    side.set_title("Land cover (ESA WorldCover 2021)", fontsize=FONT["legend"], loc="left")

    rc, rb = inp.recall_lc_vh.get("cropland", np.nan), inp.recall_lc_vh.get("built_up", np.nan)
    ax.set_title(f"Fujisawa / Shimokajiya, VH detection\n"
                 f"Flooded cropland recall {rc:.1%}   ·   flooded built-up recall {rb:.1%}",
                 fontsize=FONT["title"], loc="left")
    handles = _hit_missed_false_handles() + [
        _gsi_handle(),
        Line2D([0], [0], color="white", linewidth=1.3, linestyle=(0, (4, 2)),
               path_effects=[pe.Stroke(linewidth=2.6, foreground=BUILT_UP), pe.Normal()],
               label="Built-up area (WorldCover)"),
        Patch(facecolor=CROPLAND, label="Cropland (side map)"),
        Patch(facecolor=BUILT_UP, label="Built-up (side map)"),
        Patch(facecolor=WATER_LC, label="Water / wetland (side map)"),
        Patch(facecolor=OTHER_LC, edgecolor="#BBBBBB", label="Other (side map)"),
    ]
    side.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.04), frameon=False,
                fontsize=FONT["legend"] - 1)
    _credit(ax, f"{inp.basemap.credit}. Land cover: ESA WorldCover 2021 (CC BY 4.0). "
                 "Recall values are for the whole evaluation area, not only this window.")
    return _save(fig, out_dir / "landcover_zoom.png")


def make_all(cfg: dict, overwrite_basemap: bool = False) -> list[Path]:
    """Write every slide figure to outputs/figures/ and return their paths."""
    inp = _Inputs(cfg, overwrite_basemap)
    out_dir = inp.out / "figures"
    paths = sar_before_after(inp, out_dir)
    paths.append(flood_map_slide(inp, out_dir))
    paths.append(landcover_zoom(inp, out_dir))
    has_both = {"VH", "VV"} <= set(inp.detected)
    if has_both:
        paths.append(vv_vs_vh(inp, out_dir))
    else:
        print("  skipped vv_vs_vh_overlay.png (needs both VH and VV results)")
    if "VH" in inp.detected:
        paths.append(slide07_accuracy(inp, out_dir))
        paths.append(slide08_landcover(inp, out_dir))
    if has_both:
        paths.append(slide09_vv_vs_vh(inp, out_dir))
    paths += slide06_sar(inp, out_dir)
    return paths


# ---------------------------------------------------------------- slide-frame versions
# Full-bleed maps sized to the image frames in the portfolio deck (1920 x 1080 slides),
# rendered at 2x so they stay sharp. Legend sits in a white box inside the map, as in
# the earlier GEE figures; the slide itself carries the title and headline numbers.
SLIDE_SCALE = 2
SLIDE_FRAMES = {
    # name: (frame width px, frame height px on the slide)
    "slide07_accuracy_VH.png": (816, 524),
    "slide08_landcover_VH.png": (816, 520),
    "slide09_vv_vs_vh.png": (750, 550),
    "slide06_sar_before_VH.png": (846, 280),
    "slide06_sar_after_VH.png": (846, 280),
}
SLIDE_FLOOD_MARGIN_M = 600
SLIDE07_CENTRE = (362_250, 4_221_400)   # coast to Shimokajiya, Ebie loop upper left
SLIDE07_WIDTH_M = 6_900
SLIDE08_CENTRE = (363_800, 4_220_250)   # town right of centre, legend over cropland
SLIDE08_WIDTH_M = 2_800
SLIDE06_DB_RANGE = (-25, 0)    # same fixed grey range as the GEE layers (min -25, max 0 dB)
SLIDE06_BOUNDS = (357_100, 4_219_150, 370_520, 4_223_590)   # full study-area width, coast to hills, as in the GEE strips


def _fit_aspect(bounds, aspect: float, grid: Grid) -> tuple[float, float, float, float]:
    """Grow ``bounds`` to width/height = ``aspect`` around its centre, kept inside the grid."""
    left, bottom, right, top = bounds
    w, h = right - left, top - bottom
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (left + right) / 2, (bottom + top) / 2
    g_left, g_bottom, g_right, g_top = grid.bounds
    cx = min(max(cx, g_left + w / 2), g_right - w / 2)
    cy = min(max(cy, g_bottom + h / 2), g_top - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _slide_figure(name: str):
    w_px, h_px = SLIDE_FRAMES[name]
    fig = plt.figure(figsize=(w_px * SLIDE_SCALE / DPI, h_px * SLIDE_SCALE / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax, w_px / h_px


def _slide_legend(ax, handles, loc: str) -> None:
    leg = ax.legend(handles=handles, loc=loc, title="Flood detection legend", frameon=True, fancybox=False,
                    framealpha=0.94, edgecolor="#999999", fontsize=10, title_fontsize=11,
                    borderpad=0.7, labelspacing=0.45, handlelength=1.6, handleheight=1.0)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"


def _slide_credit(ax, text: str) -> None:
    ax.text(0.995, 0.005, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
            color="#222222", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
            zorder=30)


def _slide_save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")   # no tight bbox: keep the exact frame size
    plt.close(fig)
    return path


def _slide_map(inp: _Inputs, ax, bounds) -> tuple:
    rows, cols, extent = _window(inp.grid, bounds)
    ax.imshow(inp.basemap.rgb[rows, cols], extent=extent, interpolation="bilinear")
    return rows, cols, extent


def slide07_accuracy(inp: _Inputs, out_dir: Path) -> Path:
    name = "slide07_accuracy_VH.png"
    fig, ax, aspect = _slide_figure(name)
    cx, cy = SLIDE07_CENTRE
    w = SLIDE07_WIDTH_M
    bounds = _fit_aspect((cx - w / 2, cy - w / 4, cx + w / 2, cy + w / 4), aspect, inp.grid)
    rows, cols, extent = _slide_map(inp, ax, bounds)
    cat = _categories(inp, "VH")
    ax.imshow(_overlay([(cat["false"][rows, cols], FALSE_ALARM), (cat["missed"][rows, cols], MISSED),
                        (cat["hit"][rows, cols], HIT)]), extent=extent, interpolation="nearest")
    _finish_map(ax, extent)
    handles = [
        Patch(facecolor=HIT, alpha=0.8, label="SAR VH and GSI"),
        Patch(facecolor=MISSED, alpha=0.8, label="GSI only"),
        Patch(facecolor=FALSE_ALARM, alpha=0.8, label="SAR VH only"),
    ]
    _slide_legend(ax, handles, "lower left")
    _slide_credit(ax, inp.basemap.credit)
    return _slide_save(fig, out_dir / name)


def slide08_landcover(inp: _Inputs, out_dir: Path) -> Path:
    name = "slide08_landcover_VH.png"
    fig, ax, aspect = _slide_figure(name)
    cx, cy = SLIDE08_CENTRE
    w = SLIDE08_WIDTH_M
    bounds = _fit_aspect((cx - w / 2, cy - w / 4, cx + w / 2, cy + w / 4), aspect, inp.grid)
    rows, cols, extent = _slide_map(inp, ax, bounds)
    cat = _categories(inp, "VH")
    ax.imshow(_overlay([(cat["false"][rows, cols], FALSE_ALARM), (cat["missed"][rows, cols], MISSED),
                        (cat["hit"][rows, cols], HIT)]), extent=extent, interpolation="nearest")

    from scipy import ndimage

    sub = _subgrid(inp.grid, rows, cols)
    lc = inp.landcover[rows, cols]
    built = ndimage.binary_closing(lc == 50, structure=np.ones((3, 3)), iterations=2)
    built = remove_small_patches(built, min_pixels=150)
    outline = polygonize(built, sub, min_area_m2=20_000)
    built_style = {"color": "white", "linewidth": 1.1, "linestyle": (0, (4, 2)),
                   "path_effects": [pe.Stroke(linewidth=2.3, foreground=BUILT_UP), pe.Normal()]}
    if len(outline):
        outline["geometry"] = outline.buffer(10).buffer(-10).simplify(15)
        outline.boundary.plot(ax=ax, **built_style)
    _finish_map(ax, extent)

    handles = [
        Patch(facecolor=HIT, alpha=0.8, label="SAR VH and GSI"),
        Patch(facecolor=MISSED, alpha=0.8, label="GSI only"),
        Patch(facecolor=FALSE_ALARM, alpha=0.8, label="SAR VH only"),
        Line2D([0], [0], label="Built-up area (ESA WorldCover)", **built_style),
    ]
    _slide_legend(ax, handles, "lower left")
    _slide_credit(ax, f"{inp.basemap.credit}. Land cover: ESA WorldCover 2021")
    return _slide_save(fig, out_dir / name)


def slide09_vv_vs_vh(inp: _Inputs, out_dir: Path) -> Path:
    name = "slide09_vv_vs_vh.png"
    fig, ax, aspect = _slide_figure(name)
    bounds = _fit_aspect(_flood_bounds(inp.reference, inp.grid, SLIDE_FLOOD_MARGIN_M), aspect, inp.grid)
    rows, cols, extent = _slide_map(inp, ax, bounds)
    vh, vv, ref = inp.detected["VH"], inp.detected["VV"], inp.reference
    ax.imshow(_overlay([((ref & ~vh & ~vv)[rows, cols], MISSED), ((vh & ~vv)[rows, cols], VH_ONLY),
                        ((vv & ~vh)[rows, cols], VV_ONLY), ((vh & vv)[rows, cols], BOTH)]),
              extent=extent, interpolation="nearest")
    _finish_map(ax, extent)
    handles = [
        Patch(facecolor=BOTH, alpha=0.8, label="Detected by VH and VV"),
        Patch(facecolor=VH_ONLY, alpha=0.8, label="VH only"),
        Patch(facecolor=VV_ONLY, alpha=0.8, label="VV only"),
        Patch(facecolor=MISSED, alpha=0.8, label="GSI flood missed by both"),
    ]
    _slide_legend(ax, handles, "lower left")
    _slide_credit(ax, inp.basemap.credit)
    return _slide_save(fig, out_dir / name)


def slide06_sar(inp: _Inputs, out_dir: Path) -> list[Path]:
    """
    Raw (unfiltered) before/after VH strips for the overview slide, shown with
    the same fixed grey range as the GEE version (-25 to 0 dB): land sits in the
    dark mid-greys, so water reads black and buildings bright, and the flood's
    darkening stands out. Framed on the river mouth and the flooded plain.
    """
    vmin, vmax = SLIDE06_DB_RANGE
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")

    paths = []
    for epoch in ("before", "after"):
        name = f"slide06_sar_{epoch}_VH.png"
        fig, ax, aspect = _slide_figure(name)
        rows, cols, extent = _window(inp.grid, _fit_aspect(SLIDE06_BOUNDS, aspect, inp.grid))
        ax.imshow(inp.vh_db(epoch)[rows, cols], cmap=gray, vmin=vmin, vmax=vmax, extent=extent,
                  interpolation="nearest")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()
        _scale_bar(ax, extent)
        _slide_credit(ax, f"Contains modified Copernicus Sentinel data 2022 · VH {vmin} to {vmax} dB")
        paths.append(_slide_save(fig, out_dir / name))
    return paths
