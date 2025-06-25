"""
fetch.py – Directional wind/wave fetch on a binary land-water raster.

Author : Juliette & ChatGPT
Date   : 2025-06-25
Licence: 

Usage
-----
python fetch.py water_mask.tif fetch_out.tif --dirs 0 45 90 135 180 225 270 315
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
# from tqdm import tqdm
from numba import njit, prange


# ------------------------------------------------------------
# Helper: step vector (row, col) per direction in pixel units
# ------------------------------------------------------------
def direction_step(az_deg: float) -> tuple[float, float]:
    """
    Returns (d_row, d_col) for 1-pixel move in given azimuth.
    Azimuth is clockwise from North (0° = North, 90° = East).
    Row grows southward, so d_row =  cos(az), d_col =  sin(az).
    """
    rad = math.radians(az_deg)
    return math.cos(rad), math.sin(rad)

@njit(parallel=True, fastmath=True)
def fetch_cardinals(mask, cell_x, cell_y):
    # mask: True = water
    rows, cols = mask.shape
    f_n = np.zeros_like(mask, np.float32)
    f_s = np.zeros_like(mask, np.float32)
    f_w = np.zeros_like(mask, np.float32)
    f_e = np.zeros_like(mask, np.float32)

    # Nord (0°) : top→bottom
    for c in prange(cols):
        dist = 0.0
        for r in range(rows):
            if mask[r, c]:
                f_n[r, c] = dist
                dist += cell_y
            else:
                dist = 0.0

    # Sud (180°) : bottom→top
    for c in prange(cols):
        dist = 0.0
        for r in range(rows - 1, -1, -1):
            if mask[r, c]:
                f_s[r, c] = dist
                dist += cell_y
            else:
                dist = 0.0

    # Ouest (270°) : left→right
    for r in prange(rows):
        dist = 0.0
        for c in range(cols):
            if mask[r, c]:
                f_w[r, c] = dist
                dist += cell_x
            else:
                dist = 0.0

    # Est (90°) : right→left
    for r in prange(rows):
        dist = 0.0
        for c in range(cols - 1, -1, -1):
            if mask[r, c]:
                f_e[r, c] = dist
                dist += cell_x
            else:
                dist = 0.0

    return np.stack([f_n, f_e, f_s, f_w], axis=0)

# ------------------------------------------------------------
# Core computation
# ------------------------------------------------------------
# def compute_fetch(mask: np.ndarray,
#                   cellsize_x: float,
#                   cellsize_y: float,
#                   azimuths: list[float]
#                   ) -> np.ndarray:
#     """
#     mask         : boolean array – True for water, False for land
#     cellsize_x/y : pixel size in metres (positive)
#     azimuths     : list of directions (deg)
#     Returns 3-D array (nDir, rows, cols) with fetch lengths in metres.
#     """
#     rows, cols = mask.shape
#     n_dir = len(azimuths)
#     fetch = np.zeros((n_dir, rows, cols), dtype=np.float32)

#     # Pre-compute step vectors in pixel units
#     steps = [direction_step(az) for az in azimuths]

#     # For each direction, scan every water cell
#     for k, (az, (dr, dc)) in enumerate(zip(azimuths, steps)):
#         print(f"Direction {az}°")
#         for r in tqdm(range(rows), leave=False):
#             for c in range(cols):
#                 if not mask[r, c]:
#                     continue  # land → fetch = 0
#                 rr, cc = r, c
#                 dist = 0.0
#                 while True:
#                     rr += dr
#                     cc += dc
#                     # Stop if we leave raster
#                     if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
#                         break
#                     # Bresenham-ish: we must hit pixel centres – round indices
#                     ri, ci = int(round(rr)), int(round(cc))
#                     if not mask[ri, ci]:
#                         break  # hit land
#                     dist += math.hypot(dr * cellsize_y, dc * cellsize_x)
#                 fetch[k, r, c] = dist

#     return fetch


@njit(parallel=True, fastmath=True)
def compute_fetch(mask, cell_x, cell_y, azimuths):
    rows, cols = mask.shape
    n_dir = len(azimuths)
    fetch = np.zeros((n_dir, rows, cols), np.float32)

    # pré-calcul des pas en pixels
    dr = np.cos(np.deg2rad(azimuths))
    dc = np.sin(np.deg2rad(azimuths))
    step_len = np.hypot(dr * cell_y, dc * cell_x)

    for k in prange(n_dir):                # parallèle sur les directions
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                rr = r
                cc = c
                dist = 0.0
                while True:
                    rr += dr[k]
                    cc += dc[k]
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        break
                    ri = int(round(rr))
                    ci = int(round(cc))
                    if not mask[ri, ci]:
                        break
                    dist += step_len[k]
                fetch[k, r, c] = dist
    return fetch
# ------------------------------------------------------------
# Main wrapper
# ------------------------------------------------------------
def main():
    # parser = argparse.ArgumentParser(
    #     description="Compute directional fetch over a binary land-water raster.")
    # parser.add_argument("in_raster", help="Binary water mask raster (1 = water, 0 = land)")
    # parser.add_argument("out_raster",
    #                     help="Output GeoTIFF; one band per direction or single-band max")
    # parser.add_argument("--dirs", nargs="+", type=float, default=[0, 45, 90, 135, 180, 225, 270, 315],
    #                     help="Azimuths in degrees (default 8 compass dirs)")
    # parser.add_argument("--maxonly", action="store_true",
    #                     help="Write only the maximum fetch (single band)")
    # args = parser.parse_args()
    
    # Appel manuel (remplace les chemins ci-dessous)
    in_raster = "land_30m.tif"
    out_raster = "fetch_out.tif"
    dirs = [0, 45, 90, 135, 180, 225, 270, 315]
    maxonly = False

    in_path = Path(in_raster)
    out_path = Path(out_raster)
    


    # ---------------- open raster ----------------
    with rasterio.open(in_path) as src:
        data = src.read(1, out_dtype="uint8", resampling=Resampling.nearest)
        mask = data == 1

        # Pixel dimensions in metres (assumes projected CRS)
        a: Affine = src.transform
        cell_x = abs(a.a)
        cell_y = abs(a.e)
        profile = src.profile

    if not profile["crs"].is_projected:
        print("⚠️  CRS is geographic (degrees). Reproject to a metre CRS first.",
              file=sys.stderr)
        sys.exit(1)

    # ---------------- compute ----------------
    fetch_arr = compute_fetch(mask, cell_x, cell_y, dirs)

    # Reduce to max if requested
    if maxonly:
        fetch_arr = fetch_arr.max(axis=0, keepdims=True)

    # ---------------- save ----------------
    profile.update({
        "driver": "GTiff",
        "count": fetch_arr.shape[0],
        "dtype": "float32",
        "compress": "lzw",
        "nodata": 0
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(fetch_arr.shape[0]):
            dst.write(fetch_arr[i, :, :], i + 1)

    print(f"✅ Fetch raster written to {out_path}")


if __name__ == "__main__":
    main()
