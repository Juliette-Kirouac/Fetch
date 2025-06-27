import math
import csv
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union


def rowcol_to_xy(transform, row, col):
    """
    Convert raster row/col indices to map coordinates using the affine transform.

    Parameters:
        transform: affine transform of the raster
        row, col: integer row/column indices (0-based)

    Returns:
        (x, y): coordinates in the raster CRS
    """
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y


def get_raster_origin(raster_path, origin_rowcol=None, origin_lonlat=None):
    """
    Determine origin location and metadata for raster-based fetch.

    - Reads raster transform, CRS, and cell sizes.
    - Converts either (row, col) or (lon, lat in EPSG:4326) to:
        * origin_xy: map coords in raster CRS
        * row, col: pixel indices
    - Validates origin lies within bounds and on water (mask==0).

    Returns:
        dict with keys:
            'origin_xy', 'row', 'col', 'cell_x', 'cell_y', 'crs'
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        cell_x = abs(transform.a)
        cell_y = abs(transform.e)

        # Determine origin coordinates and pixel indices
        if origin_lonlat is not None:
            # Reproject input lon/lat (EPSG:4326) to raster CRS if needed
            lon, lat = origin_lonlat
            if crs.is_geographic:
                x, y = lon, lat
            else:
                xs, ys = rio_transform('EPSG:4326', crs, [lon], [lat])
                x, y = xs[0], ys[0]
            # Bounds check
            minx, miny, maxx, maxy = src.bounds
            if not (minx <= x <= maxx and miny <= y <= maxy):
                raise ValueError('Origin lies outside raster bounds.')
            row, col = src.index(x, y)
        elif origin_rowcol is not None:
            row, col = origin_rowcol
            # Dimensions check
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError('Origin row/col lie outside raster dimensions.')
            x, y = rowcol_to_xy(transform, row, col)
        else:
            raise ValueError('Raster mode requires origin_rowcol or origin_lonlat')

        # Water check: 0=water, 1=land
        val = src.read(1, window=Window(col, row, 1, 1), out_dtype='uint8')[0, 0]
        if val == 1:
            raise ValueError('Origin lies on land (mask==1).')

    return {
        'origin_xy': (x, y),
        'row': row,
        'col': col,
        'cell_x': cell_x,
        'cell_y': cell_y,
        'crs': crs
    }


def compute_fetch_dynamic(raster_path, cell_x, cell_y, row, col, azimuths):
    """
    Compute directional fetch on a raster mask without loading entire array into memory.

    Reads one pixel at a time along each azimuth until land (value==1) or raster edge.

    Parameters:
        raster_path: path to mask raster (0=water,1=land)
        cell_x, cell_y: pixel size in CRS units
        row, col: origin pixel indices
        azimuths: iterable of angles in degrees

    Returns:
        dict mapping each azimuth -> fetch distance (meters)
    """
    results = {}
    with rasterio.open(raster_path) as src:
        for az in azimuths:
            rad = math.radians(az)
            dr, dc = -math.cos(rad), math.sin(rad)
            step_len = math.hypot(dr * cell_y, dc * cell_x)
            dist = 0.0
            rr, cc = float(row), float(col)

            while True:
                rr += dr
                cc += dc
                # Exit if outside raster
                if not (0 <= rr < src.height and 0 <= cc < src.width):
                    break
                ri, ci = int(round(rr)), int(round(cc))
                # Read single value
                val = src.read(1, window=Window(ci, ri, 1, 1), out_dtype='uint8')[0, 0]
                # Stop at land
                if val == 1:
                    break
                dist += step_len

            results[az] = dist
    return results


def get_vector_origin(vector_path, origin_lonlat=None):
    """
    Determine origin location and metadata for vector-based fetch.

    - Reads coastline vector (supports .shp, .gpkg, etc.).
    - Assumes input lon/lat in EPSG:4326; reprojects to vector CRS if needed.
    - Validates origin within layer bounds and on water (outside polygons).

    Returns:
        origin_xy: (x, y) in vector CRS
        coast_union: unified geometry for intersection tests
        crs: CRS of the vector layer
    """
    if origin_lonlat is None:
        raise ValueError('Vector mode requires origin_lonlat (lon, lat) in EPSG:4326')

    # Load vector (auto-detect .gpkg layers)
    import fiona
    ext = Path(vector_path).suffix.lower()
    if ext == '.gpkg':
        layers = fiona.listlayers(vector_path)
        coast = gpd.read_file(vector_path, layer=layers[0])
    else:
        coast = gpd.read_file(vector_path)

    crs = coast.crs
    # Reproject input point
    lon, lat = origin_lonlat
    origin_pt_geo = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326')
    if crs.is_geographic:
        origin_xy = (lon, lat)
    else:
        origin_transformed = origin_pt_geo.to_crs(crs)
        origin_xy = tuple(origin_transformed.geometry.iloc[0].coords)[0]

    origin_pt = Point(origin_xy)
    # Bounds check
    minx, miny, maxx, maxy = coast.total_bounds
    x, y = origin_xy
    if not (minx <= x <= maxx and miny <= y <= maxy):
        raise ValueError('Origin lies outside vector bounds.')

    # Land check: assume polygons = land
    land_polys = coast[coast.geom_type.isin(['Polygon', 'MultiPolygon'])]
    if not land_polys.empty:
        land_union = unary_union(land_polys.geometry)
        if land_union.contains(origin_pt):
            raise ValueError('Vector origin lies within land polygon.')

    # Unified geometry for ray intersections
    coast_union = unary_union(coast.geometry)
    return origin_xy, coast_union, crs


def compute_fetch_from_shapefile(coast_union, origin_xy, max_dist, azimuths):
    """
    Compute directional fetch using a unified coastline geometry.

    For each azimuth, casts a ray of length max_dist and finds first intersection.
    Returns dict mapping azimuth -> fetch distance.
    """
    origin_pt = Point(origin_xy)
    results = {}

    for az in azimuths:
        rad = math.radians(az)
        end_pt = (
            origin_xy[0] + math.sin(rad) * max_dist,
            origin_xy[1] + math.cos(rad) * max_dist
        )
        ray = LineString([origin_xy, end_pt])
        inter = ray.intersection(coast_union)
        if inter.is_empty:
            results[az] = max_dist
        else:
            pts = list(inter.geoms) if hasattr(inter, 'geoms') else [inter]
            results[az] = min(origin_pt.distance(pt) for pt in pts)

    return results


def get_azimuths(step=None, angles_list=None):
    """
    Return an array of azimuths.
    Provide either a uniform step (e.g. 2°) or an explicit list, not both.
    """
    if step is not None and angles_list is not None:
        raise ValueError('Specify either step or angles_list, not both.')
    if angles_list is not None:
        return np.array(sorted(angles_list), dtype=float)
    if step is not None:
        return np.arange(0.0, 360.0, step, dtype=float)
    raise ValueError('Either step or angles_list must be provided.')


def warn_on_jumps(fetch_dict, threshold):
    """
    Print warnings if adjacent fetch distances differ by more than threshold.
    """
    azs = sorted(fetch_dict)
    vals = [fetch_dict[az] for az in azs]
    for i in range(len(vals)):
        delta = abs(vals[(i+1) % len(vals)] - vals[i])
        if delta > threshold:
            print(f"⚠️ Jump of {delta:.1f} m between {azs[i]:.1f}° and {azs[(i+1)%len(vals)]:.1f}°")
    overall = max(vals) - min(vals)
    if overall > threshold:
        print(f"⚠️ Overall fetch range {overall:.1f} m exceeds {threshold:.1f} m")


def export_shapefiles(fetch_dict, origin_xy, crs, out_lines, out_env):
    """
    Write individual fetch lines and the envelope polygon to Shapefiles.
    """
    records, ends = [], []
    for az, dist in sorted(fetch_dict.items()):
        rad = math.radians(az)
        end = (origin_xy[0] + math.sin(rad)*dist,
            origin_xy[1] + math.cos(rad)*dist)
        ends.append(end)
        records.append({'azimuth': az, 'distance_m': dist,'geometry': LineString([origin_xy, end])})
    gpd.GeoDataFrame(records, crs=crs).to_file(out_lines, driver='ESRI Shapefile')

    envelope = Polygon(ends + [ends[0]])
    gpd.GeoDataFrame([{'geometry': envelope}], crs=crs).to_file(out_env, driver='ESRI Shapefile')


def export_csv(fetch_dict, output_csv):
    """
    Write fetch results to a simple CSV (azimuth, distance).
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['azimuth_deg', 'distance_m'])
        for az in sorted(fetch_dict):
            writer.writerow([az, fetch_dict[az]])


def run_fetch(use_vector, raster_path=None, vector_path=None,
              origin_rowcol=None, origin_lonlat=None,
              max_dist=None, step=None, angles_list=None,
              warn_thresh=0, out_lines=None, out_env=None, out_csv=None):
    """
    Unified function to compute and output fetch results.
    Returns the fetch_dict and origin_xy.
    """
    azs=get_azimuths(step,angles_list)
    if use_vector:
        origin_xy, coast_union, crs = get_vector_origin(vector_path, origin_lonlat)
        fetch=compute_fetch_from_shapefile(coast_union,origin_xy,max_dist,azs)
    else:
        ras=get_raster_origin(raster_path, origin_rowcol, origin_lonlat)
        origin_xy=ras['origin_xy']; crs=ras['crs']
        fetch=compute_fetch_dynamic(raster_path, ras['cell_x'],ras['cell_y'],ras['row'],ras['col'],azs)
    if warn_thresh>0: warn_on_jumps(fetch,warn_thresh)
    if out_lines and out_env: export_shapefiles(fetch,origin_xy,crs,out_lines,out_env)
    if out_csv: export_csv(fetch,out_csv)
    return fetch, origin_xy
