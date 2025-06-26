"""
fetch_point.py – Compute directional fetch from a single point.

Usage examples:
  # Using geographic coordinates (in the raster's CRS):
  python fetch_point.py \
    --in land_30m.tif \
    --lon -73.5 --lat 45.45 \
    --step 2

  # Using pixel indices directly:
  python fetch_point.py \
    --in land_30m.tif \
    --row 1500 --col 820 \
    --step 2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


def compute_fetch_at_point(mask, cell_x, cell_y, row, col, azimuths):
    results = {}
    for az in azimuths:
        # step vector in pixel‐units
        rad = math.radians(az)
        dr  = -math.cos(rad)   # → 0° gives dr = -1 (moves up/north)
        dc  =  math.sin(rad)   # → 90° gives dc = +1 (moves right/east)
        step_len = math.hypot(dr * cell_y, dc * cell_x)

        dist = 0.0
        rr, cc = float(row), float(col)
        # march until land or edge
        while True:
            rr += dr
            cc += dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                break
            ri, ci = int(round(rr)), int(round(cc))
            # mask True = water; stop when hit land
            if not mask[ri, ci]:
                break
            dist += step_len

        results[az] = dist
    return results


def main():
    p = argparse.ArgumentParser(
        description="Compute fetch for a single point in a binary water/land raster.")
    # p.add_argument("--in",       dest="in_raster", required=True,
    #                help="Input GeoTIFF (0=water, 1=land)")
    # p.add_argument("--lon",      type=float, help="X coordinate in raster CRS")
    # p.add_argument("--lat",      type=float, help="Y coordinate in raster CRS")
    # p.add_argument("--row",      type=int,   help="Row index (0-based)")
    # p.add_argument("--col",      type=int,   help="Col index (0-based)")
    # p.add_argument("--step",     type=float, default=2.0,
    #                help="Angular step in degrees (default 2°)")
    # args = p.parse_args()
    
    in_raster = "land_30m.tif"       # votre fichier
    step = 2                         # degrés de rotation entre les fetch calculés
    row = 2723                       # Row index (0-based)
    col = 1403                       # Col index (0-based)
    lon = None
    lat = None
    
    # 0 : sud, 90: est, 180 : nord, 270 : ouest
    # open raster and build water mask
    with rasterio.open(in_raster) as src:
        arr = src.read(1, out_dtype="uint8", resampling=Resampling.nearest)
        mask = (arr == 0)  # True = water
        tr = src.transform
        cell_x = abs(tr.a)
        cell_y = abs(tr.e)

        # determine row/col
        if lon is not None and lat is not None:
            row, col = src.index(lon, lat)
        elif row is not None and col is not None:
            row, col = row, col
        else:
            p.error("You must provide either --lon/--lat or --row/--col.")

    # build azimuth list
    azs = np.arange(0.0, 360.0, step, dtype=np.float32)

    # compute
    fetchs = compute_fetch_at_point(mask, cell_x, cell_y, row, col, azs)

    # output
    print(f"Fetch distances from point (row={row}, col={col}):\n")
    for az in sorted(fetchs):
        print(f"  {az:6.1f}° → {fetchs[az]:8.1f} m")


if __name__ == "__main__":
    main()
