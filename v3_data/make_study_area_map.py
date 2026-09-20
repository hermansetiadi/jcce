"""
Two-site study-area locator map for the manuscript (Figure: study area).

Panel (a): Indonesia / SE-Asia context with a box marking the Java zoom.
Panel (b): Java zoom with the two AOIs:
    - Sepatan Timur, Tangerang Regency (Banten)   ~ 106.575 E, -6.135 S   [denser landscape]
    - Situbondo (East Java)                        ~ 114.009 E, -7.706 S   [sparser landscape]

Coastlines come from Natural Earth (downloaded + cached on first run). If the
download fails (offline), the script falls back to a clean schematic with no
coastline, clearly labelled.

!! COORDINATES ARE APPROXIMATE regency/town centroids. Replace `AOIS` with the
   exact AOI bounding boxes (or paste a GeoJSON of the tile footprints) before
   final submission.

Writes to figures/ :
    study_area_map.{png,pdf}

Run:
    python make_study_area_map.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
CACHE = FIG_DIR / "_geocache"
FIG_DIR.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# --- AOI locations (lon, lat) ---
# Tangerang: EXACT tile-footprint centroid + bounds derived from the dataset's
#            qgis_georef VRTs (305 tiles, EPSG:4326):
#            lon [106.5821, 106.6352], lat [-6.1474, -6.0967].
# Situbondo: approximate site centroid (no georeferenced tiles available).
# lab_dx/lab_dy = label offset from marker (deg); lab_ha = label horizontal anchor.
AOIS = [
    {"name": "Tangerang /\nSepatan Timur", "lon": 106.6087, "lat": -6.1220,
     "note": "Banten · denser", "color": "#0072B2",
     "bbox": (106.5821, 106.6352, -6.1474, -6.0967),
     "lab_dx": 0.3, "lab_dy": -1.25, "lab_ha": "left"},
    {"name": "Situbondo", "lon": 114.009, "lat": -7.706,
     "note": "East Java · sparser", "color": "#D55E00",
     "lab_dx": -0.3, "lab_dy": 0.55, "lab_ha": "right"},
]

INDO_EXTENT = (94, 142, -11.5, 7.0)   # lon_min, lon_max, lat_min, lat_max
JAVA_EXTENT = (105.0, 115.5, -9.0, -5.3)

NE_URLS = [
    ("ne_50m_admin_0_countries", "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"),
    ("ne_110m_admin_0_countries", "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"),
]


def get_countries():
    """Return a GeoDataFrame of country polygons, or None if unavailable."""
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"  [warn] geopandas unavailable ({e}); schematic fallback.")
        return None
    for name, url in NE_URLS:
        zp = CACHE / f"{name}.zip"
        try:
            if not zp.exists():
                print(f"  downloading {name} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r, open(zp, "wb") as f:
                    f.write(r.read())
            gdf = gpd.read_file(f"zip://{zp}")
            print(f"  using {name} ({len(gdf)} features)")
            return gdf
        except Exception as e:
            print(f"  [warn] {name} failed: {type(e).__name__} {str(e)[:80]}")
            continue
    print("  [warn] no Natural Earth data; schematic fallback.")
    return None


def _frame(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linestyle=":")


def draw_basemap(ax, gdf, extent, land="#eae6df", edge="#9a9a9a", sea="#dfeaf2"):
    ax.set_facecolor(sea)
    if gdf is not None:
        gdf.plot(ax=ax, color=land, edgecolor=edge, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "(coastline unavailable —\nschematic only)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#999")
    _frame(ax, extent)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    gdf = get_countries()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- (a) Indonesia context ----
    draw_basemap(ax1, gdf, INDO_EXTENT)
    jx0, jx1, jy0, jy1 = JAVA_EXTENT
    ax1.add_patch(Rectangle((jx0, jy0), jx1 - jx0, jy1 - jy0,
                            fill=False, edgecolor="#d62728", linewidth=1.8, zorder=5))
    ax1.annotate("Java (see panel b)", (jx1, jy1), xytext=(jx1 + 1, jy1 + 1.5),
                 fontsize=8, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728"))
    for a in AOIS:
        ax1.plot(a["lon"], a["lat"], "*", ms=9, color=a["color"],
                 mec="black", mew=0.4, zorder=6)
    ax1.set_title("(a) Study region — Indonesia")

    # ---- (b) Java zoom with AOIs ----
    draw_basemap(ax2, gdf, JAVA_EXTENT)
    for a in AOIS:
        if "bbox" in a:  # exact tile-footprint extent (tiny at this scale)
            x0, x1, y0, y1 = a["bbox"]
            ax2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, edgecolor=a["color"],
                                    linewidth=1.0, zorder=5))
        ax2.plot(a["lon"], a["lat"], "*", ms=18, color=a["color"],
                 mec="black", mew=0.6, zorder=6)
        txt = ax2.annotate(f"{a['name']}\n({a['note']})",
                           (a["lon"], a["lat"]),
                           xytext=(a["lon"] + a["lab_dx"], a["lat"] + a["lab_dy"]),
                           fontsize=8.5, fontweight="bold", color="black",
                           ha=a["lab_ha"], zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax2.set_title("(b) Java — paddy AOIs", pad=12)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"study_area_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures/study_area_map.png and .pdf")


if __name__ == "__main__":
    main()
