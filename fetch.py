import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

def compute_fetch_at_point(
    mask: np.ndarray,
    cell_x: float,
    cell_y: float,
    row: int,
    col: int,
    azimuths: list[float]
) -> dict[float, float]:
    """
    mask       : bool array, True=water, False=land
    cell_x/y   : taille de pixel en m (assume CRS projeté)
    row, col   : indices du point d'intérêt (0-index)
    azimuths   : liste d'azimuts (°) depuis le Nord
    return     : dict {az: distance en m}
    """
    results = {}
    for az in azimuths:
        # vecteur pas en pixels
        rad = math.radians(az)
        dr = math.cos(rad)
        dc = math.sin(rad)
        step_len = math.hypot(dr * cell_y, dc * cell_x)

        # init
        dist = 0.0
        rr, cc = row, col

        # avance jusqu'à terre ou bord
        while True:
            rr += dr
            cc += dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                break
            ri, ci = int(round(rr)), int(round(cc))
            if not mask[ri, ci]:  # si c'est de la terre
                break
            dist += step_len

        results[az] = dist

    return results


if __name__ == "__main__":
    # --- Paramètres à ajuster ---
    in_raster = "land_30m.tif"       # votre fichier
    # Coordonnées du point d'intérêt en indices de pixel
    row = 2724                       # ex: ligne (Y) du raster
    col =  1405                       # ex: colonne (X) du raster

    # Directions souhaitées
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315] # 0 : north, 90 : east, 180 : south, 270 : west
    

    # -----------------------------

    # Lecture binaire et inversion
    with rasterio.open(in_raster) as src:
        arr = src.read(1, out_dtype="uint8", resampling=Resampling.nearest)
        # ici 0 = eau, 1 = terre → mask_water est True pour l'eau
        mask_water = (arr == 0)
        a = src.transform
        cell_x = abs(a.a)
        cell_y = abs(a.e)

    # Calcul fetch pour ce point
    fetch_dict = compute_fetch_at_point(mask_water, cell_x, cell_y, row, col, azimuths)

    # Affichage
    print(f"Fetch depuis le pixel (row={row}, col={col}):")
    for az, dist in fetch_dict.items():
        print(f"  • {az:3.0f}° → {dist:.1f} m")
