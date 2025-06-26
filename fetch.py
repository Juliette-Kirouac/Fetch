"""
fetch_point_checked.py – Compute fetch from one point + warning on large jumps.

Usage:
  python fetch_point_checked.py \
    --in land_30m.tif \
    --lon -73.5 --lat 45.45 \
    --step 2 \
    --warn 100
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
        rad = math.radians(az)
        dr = -math.cos(rad)    # 0°→nord
        dc = math.sin(rad)
        step_len = math.hypot(dr * cell_y, dc * cell_x)

        dist = 0.0
        rr, cc = float(row), float(col)
        while True:
            rr += dr; cc += dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                break
            ri, ci = int(round(rr)), int(round(cc))
            if not mask[ri, ci]:
                break
            dist += step_len

        results[az] = dist
    return results


def main():
    p = argparse.ArgumentParser(
        description="Fetch for one point + warn on large directional jumps.")
    # p.add_argument("--in", dest="in_raster", required=True,
    #                help="Input GeoTIFF (0=water,1=land)")
    # p.add_argument("--lon", type=float, help="X coordinate in raster CRS")
    # p.add_argument("--lat", type=float, help="Y coordinate in raster CRS")
    # p.add_argument("--row", type=int, help="Row index (0-based)")
    # p.add_argument("--col", type=int, help="Col index (0-based)")
    # p.add_argument("--step", type=float, default=2.0,
    #                help="Angular step in degrees (default 2°)")
    # p.add_argument("--warn", type=float, default=None,
    #                help="Threshold (m) for alerting on large fetch jumps")
    # args = p.parse_args()
    
    in_raster = "land_30m_v3.tif"       # votre fichier
    step = 2                         # Angular step in degrees (default 2°)
    row = 8584                       # Row index (0-based)
    col = 1405                       # Col index (0-based)
    lon = None                       # X coordinate in raster CRS
    lat = None                       # Y coordinate in raster CRS
    warn = 10000                     # Threshold (m) for alerting on large fetch jumps

    # --- load raster and mask ---
    with rasterio.open(in_raster) as src:
        arr = src.read(1, out_dtype="uint8", resampling=Resampling.nearest)
        mask = (arr == 0)  # True = water
        tr = src.transform
        cell_x, cell_y = abs(tr.a), abs(tr.e)

        if lon is not None and lat is not None:
            row, col = src.index(lon, lat)
        elif row is not None and col is not None:
            row, col = row, col
        else:
            p.error("Provide --lon/--lat or --row/--col.")

    # --- compute fetch distances ---
    azs = np.arange(0.0, 360.0, step, dtype=np.float32)
    fetchs = compute_fetch_at_point(mask, cell_x, cell_y, row, col, azs)

    # --- output results ---
    print(f"Fetch from point (row={row}, col={col}):\n")
    sorted_az = sorted(fetchs.keys())
    values = [fetchs[az] for az in sorted_az]
    for az, dist in zip(sorted_az, values):
        print(f"  {az:6.1f}° → {dist:8.1f} m")

    # --- warning on large jumps ---
    if warn is not None:
        # check adjacent directions
        print(f"\nChecking for jumps > {warn:.1f} m...")
        # compute differences between successive directions (circular)
        diffs = []
        n = len(sorted_az)
        for i in range(n):
            d1 = values[i]
            d2 = values[(i+1) % n]
            delta = abs(d2 - d1)
            diffs.append(delta)
            if delta > warn:
                print(f" ⚠️ Jump of {delta:.1f} m between {sorted_az[i]:.1f}° and {sorted_az[(i+1)%n]:.1f}°")
        # overall max-min
        overall = max(values) - min(values)
        if overall > warn:
            print(f" ⚠️ Overall fetch range is {overall:.1f} m (max-min) exceeding {warn:.1f} m")
        else:
            print(" No jumps exceed threshold.")
    

if __name__ == "__main__":
    main()
