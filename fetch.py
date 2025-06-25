# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:29:50 2025

@author: julie
"""

#!/usr/bin/env python3
"""
fetch_numba.py – Fetch rasters accelerated with Numba (8 directions)

Usage
-----
python fetch_numba.py water_mask.tif fetch_out.tif [--maxonly]

Dependencies
------------
pip install rasterio numpy numba tqdm
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from numba import njit, prange
from tqdm import tqdm


@njit(parallel=True, fastmath=True)
def compute_fetch_numba(mask, cell_x, cell_y, azimuths):
    rows, cols = mask.shape
    n_dir = len(azimuths)
    fetch = np.zeros((n_dir, rows, cols), dtype=np.float32)

    # Pré-calcul des pas et longueurs de pas
    dr = np.cos(np.deg2rad(azimuths))
    dc = np.sin(np.deg2rad(azimuths))
    step_len = np.hypot(dr * cell_y, dc * cell_x)

    # Pour chaque direction, chaque pixel eau, on trace le fetch
    for k in prange(n_dir):
        dk_dr = dr[k]
        dk_dc = dc[k]
        lk = step_len[k]
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                rr = r
                cc = c
                dist = 0.0
                # on avance jusqu’à heurter la terre ou sortir du raster
                while True:
                    rr += dk_dr
                    cc += dk_dc
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        break
                    ri = int(rr + 0.5)
                    ci = int(cc + 0.5)
                    if not mask[ri, ci]:
                        break
                    dist += lk
                fetch[k, r, c] = dist

    return fetch


def main():
    # parser = argparse.ArgumentParser(
    #     description="Compute directional fetch (8 dirs) with Numba acceleration.")
    # parser.add_argument("in_raster",
    #                     help="Binary water mask (1=water, 0=land)")
    # parser.add_argument("out_raster",
    #                     help="Output GeoTIFF (one band per direction)")
    # parser.add_argument("--maxonly", action="store_true",
    #                     help="Export only the maximum fetch per pixel")
    # args = parser.parse_args()

    # Appel manuel (remplace les chemins ci-dessous)
    in_raster = "land_30m.tif"
    out_raster = "fetch_out.tif"
    dirs = [0, 45, 90, 135, 180, 225, 270, 315]
    maxonly = False

    in_path = Path(in_raster)
    out_path = Path(out_raster)

    # Directions standard : 0,45,...315
    azimuths = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)

    # Lire le raster
    with rasterio.open(in_path) as src:
        data = src.read(1, out_dtype="uint8", resampling=Resampling.nearest)
        mask = data == 1

        a: Affine = src.transform
        cell_x = abs(a.a)
        cell_y = abs(a.e)
        profile = src.profile

    if not profile["crs"].is_projected:
        raise ValueError("Reproject to a projected CRS in metres first.")

    # Calcul JIT (affiche un progrès global sous tqdm)
    print("🔄 Compilation et calcul du fetch …")
    # on exécute une fois pour compiler
    _ = compute_fetch_numba(mask[:2, :2], cell_x, cell_y, azimuths)  # warm-up
    # puis calcul réel
    fetch_arr = compute_fetch_numba(mask, cell_x, cell_y, azimuths)

    # Réduire en max si demandé
    if maxonly:
        fetch_arr = fetch_arr.max(axis=0, keepdims=True)

    # Préparer le profil pour multi-bandes
    profile.update({
        "driver": "GTiff",
        "count": fetch_arr.shape[0],
        "dtype": "float32",
        "compress": "lzw",
        "nodata": 0
    })

    # Écrire le GeoTIFF
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(fetch_arr.shape[0]):
            print(f"💾 Écriture de la bande {i+1}/{fetch_arr.shape[0]} …")
            dst.write(fetch_arr[i], i + 1)

    print(f"✅ Fetch raster écrit dans : {out_path}")


if __name__ == "__main__":
    main()
