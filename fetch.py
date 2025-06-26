import math
from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def compute_fetch_at_point(mask, cell_x, cell_y, row, col, azimuths):
    """
    Compute directional fetch from a single point.

    Parameters:
    - mask: 2D numpy array of bool (True=water, False=land)
    - cell_x, cell_y: pixel size in CRS units (meters)
    - row, col: origin pixel indices
    - azimuths: iterable of azimuth angles (degrees clockwise from North)

    Returns:
    - dict mapping azimuth -> fetch distance (meters)
    """
    results = {}
    for az in azimuths:
        rad = math.radians(az)
        dr = -math.cos(rad)    # 0° → north
        dc =  math.sin(rad)    # 90° → east
        step_len = math.hypot(dr * cell_y, dc * cell_x)

        dist = 0.0
        rr, cc = float(row), float(col)
        while True:
            rr += dr; cc += dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                break
            ri, ci = int(round(rr)), int(round(cc))
            if not mask[ri, ci]:  # hit land
                break
            dist += step_len
        results[az] = dist
    return results


def warn_on_jumps(fetch_dict, threshold):
    """
    Print warnings when fetch distances have large jumps.

    Parameters:
    - fetch_dict: dict of azimuth -> distance
    - threshold: float, jump threshold in meters
    """
    azs = sorted(fetch_dict.keys())
    values = [fetch_dict[az] for az in azs]
    n = len(values)
    for i in range(n):
        d1, d2 = values[i], values[(i+1) % n]
        delta = abs(d2 - d1)
        if delta > threshold:
            print(f"⚠️ Jump of {delta:.1f} m between {azs[i]:.1f}° and {azs[(i+1)%n]:.1f}°")
    overall = max(values) - min(values)
    if overall > threshold:
        print(f"⚠️ Overall fetch range {overall:.1f} m exceeds threshold {threshold:.1f} m")


def rowcol_to_xy(transform, row, col):
    """
    Convert raster row/col indices to geographic coordinates (x, y).

    Parameters:
    - transform: Affine transform from rasterio
    - row, col: integer pixel indices

    Returns:
    - tuple (x, y) in CRS units
    """
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y


def export_shapefiles(fetch_dict, origin_xy, crs, out_lines, out_envelope):
    """
    Export fetch lines and envelope as Shapefiles.

    Parameters:
    - fetch_dict: dict of azimuth -> distance
    - origin_xy: tuple (x, y) in CRS coordinates
    - crs: rasterio CRS or equivalent
    - out_lines: path to output lines shapefile (.shp)
    - out_envelope: path to output envelope shapefile (.shp)
    """
    lon, lat = origin_xy
    line_records = []
    endpoints = []
    for az in sorted(fetch_dict.keys()):
        dist = fetch_dict[az]
        rad = math.radians(az)
        dx = math.sin(rad) * dist
        dy = math.cos(rad) * dist
        end_x = lon + dx
        end_y = lat + dy
        endpoints.append((end_x, end_y))
        line = LineString([(lon, lat), (end_x, end_y)])
        line_records.append({
            'azimuth': float(az),
            'distance_m': float(dist),
            'geometry': line
        })
    # write lines
    gdf_lines = gpd.GeoDataFrame(line_records, crs=crs)
    gdf_lines.to_file(out_lines, driver='ESRI Shapefile')

    # write envelope
    ring = endpoints + [endpoints[0]]
    poly = Polygon(ring)
    gdf_env = gpd.GeoDataFrame([{'geometry': poly}], crs=crs)
    gdf_env.to_file(out_envelope, driver='ESRI Shapefile')


def export_csv(fetch_dict, output_csv):
    """
    Export fetch distances to CSV.

    Parameters:
    - fetch_dict: dict of azimuth -> distance
    - output_csv: path to output CSV file
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['azimuth_deg', 'distance_m'])
        for az in sorted(fetch_dict.keys()):
            writer.writerow([az, fetch_dict[az]])


def main():
    # Customize parameters below
    in_raster = 'land_30m_v3.tif'
    # Choose either lon/lat or row/col
    use_rowcol = True
    lon = None; lat = None
    row, col = 8584, 1405
    step = 2.0
    warn_threshold = 10000
    out_lines = 'fetch_lines.shp'
    out_envelope = 'fetch_env.shp'
    out_csv = 'fetch_results.csv'

    # load raster and mask
    with rasterio.open(in_raster) as src:
        arr = src.read(1, out_dtype='uint8', resampling=Resampling.nearest)
        mask = (arr == 0)
        transform = src.transform
        crs = src.crs
        if use_rowcol:
            x, y = rowcol_to_xy(transform, row, col)
        else:
            x, y = lon, lat
            row, col = src.index(lon, lat)
        cell_x = abs(transform.a)
        cell_y = abs(transform.e)

    # compute fetch
    azs = np.arange(0.0, 360.0, step, dtype=np.float32)
    fetch_dict = compute_fetch_at_point(mask, cell_x, cell_y, row, col, azs)

    # warn on jumps
    warn_on_jumps(fetch_dict, warn_threshold)

    # export outputs
    export_shapefiles(fetch_dict, (x, y), crs, out_lines, out_envelope)
    export_csv(fetch_dict, out_csv)

    print('Done. Files:', out_lines, out_envelope, out_csv)


if __name__ == '__main__':
    main()
