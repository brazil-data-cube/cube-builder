#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Module for Grid creation."""

import math
from typing import List, Tuple

from bdc_catalog.utils import geom_to_wkb
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import Polygon


def _create_tiles(tile_size: Tuple[float, float],
                  grid_x_min: float, grid_y_max: float,
                  bbox: Tuple[float, float, float, float],
                  srid: float):
    tile_size_x, tile_size_y = tile_size

    tiles = []
    features = []
    xmin, xmax, ymin, ymax = bbox

    h_min = int((xmin - grid_x_min) / tile_size_x)
    h_max = int((xmax - grid_x_min) / tile_size_x)
    v_min = int((grid_y_max - ymax) / tile_size_y)
    v_max = int((grid_y_max - ymin) / tile_size_y)

    for ix in range(h_min, h_max + 1):
        x1 = grid_x_min + ix * tile_size_x
        x2 = x1 + tile_size_x
        for iy in range(v_min, v_max + 1):
            y1 = grid_y_max - iy * tile_size_y
            y2 = y1 - tile_size_y

            polygon = geom_to_wkb(
                Polygon(
                    [
                        (x1, y2),
                        (x2, y2),
                        (x2, y1),
                        (x1, y1),
                        (x1, y2)
                    ]
                ),
                srid=srid
            )

            # Insert tile
            tile_name = '{0:03d}{1:03d}'.format(ix, iy)
            tiles.append(dict(
                name=tile_name
            ))
            features.append(dict(
                tile=tile_name,
                geom=polygon
            ))

    return tiles, features


def create_grids(names: List[str], projection, meridian,
                 degreesx: List[float], degreesy: List[float], bbox, srid=100001):
    """Create a list of hierarchically grids for Brazil Data Cube.

    With a meridian, bounding box and list of degrees, this function creates a list of grids
    based in the lowest degree. After that, the other grids are generated inherit the lowest
    to be hierarchical.

    For example, the grids BDC_SM_V2, BDC_MD_V2 and BDC_LG_V2 are hierarchically identical.
        - BDC_SM_V2 consists in tiles of 1.5 by 1.0 degrees.
        - BDC_MD_V2 represents 4 tiles of BDC_SM_V2 (2x2).
        - BDC_LG_V2 represents 16 tiles of BDC_SM_V2 (4x4) or 4 tiles of BDC_MD_V2 (2x2)

    Note:
        Keep the order of names according degree's

    Args:
        names (List[str]): List of grid names
        projection (str): The Grid projection kind - "sinu" for sinusoidal grids and "aea" for Albers Equal Area.
        meridian (float): The center pixel for grid as reference.
        degreesx (List[float]): List the degree in X
        degreesy (List[float]): List the degree in Y
        bbox (Tuple[float, float, float, float]): The bounding box limits (minx, miny, maxx, maxy).
        srid (int): The projection SRID to create.
    """
    bbox_obj = {
        "w": float(bbox[0]),
        "n": float(bbox[1]),
        "e": float(bbox[2]),
        "s": float(bbox[3])
    }
    tile_srs_p4 = "+proj=longlat +ellps=GRS80 +no_defs"
    if projection == 'aea':
        tile_srs_p4 = "+proj=aea +lat_0=-12 +lon_0={} +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs".format(
            meridian)
    elif projection == 'sinu':
        tile_srs_p4 = "+proj=sinu +lon_0={} +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs".format(
            meridian)

    ref_degree_x = degreesx[0]
    ref_degree_y = degreesy[0]

    # Tile size in meters (dx,dy) at center of system (argsmeridian,0.)
    src_crs = '+proj=longlat +ellps=GRS80 +no_defs'
    dst_crs = tile_srs_p4
    xs = [(meridian - ref_degree_x / 2.), (meridian + ref_degree_x / 2.), meridian, meridian, 0.]
    ys = [0., 0., -ref_degree_y / 2., ref_degree_y / 2., 0.]
    out = transform(CRS.from_proj4(src_crs), CRS.from_proj4(dst_crs), xs, ys, zs=None)
    xmin_center_tile = out[0][0]
    xmax_center_tile = out[0][1]
    ymin_center_tile = out[1][2]
    ymax_center_tile = out[1][3]
    tile_size_x = xmax_center_tile - xmin_center_tile
    tile_size_y = ymax_center_tile - ymin_center_tile

    bbox_alber_envelope_xs = [bbox_obj['w'], bbox_obj['e'], meridian, meridian]
    bbox_alber_envelope_ys = [0., 0., bbox_obj['n'], bbox_obj['s']]
    bbox_alber_envelope = transform(src_crs, dst_crs, bbox_alber_envelope_xs, bbox_alber_envelope_ys, zs=None)

    total_tiles_left = math.ceil(abs((xmin_center_tile - bbox_alber_envelope[0][0])) / tile_size_x)
    total_tiles_upper = math.ceil(abs((ymax_center_tile - bbox_alber_envelope[1][2])) / tile_size_y)

    # Border coordinates of WRS grid
    x_min = xmin_center_tile - (tile_size_x * total_tiles_left)
    y_max = ymax_center_tile + (tile_size_y * total_tiles_upper)

    # Upper Left is (xl,yu) Bottom Right is (xr,yb)
    xs = [bbox_obj['w'], bbox_obj['e'], meridian, meridian]
    ys = [0., 0., bbox_obj['n'], bbox_obj['s']]
    out = transform(src_crs, dst_crs, xs, ys, zs=None)
    xl = out[0][0]
    xr = out[0][1]
    yu = out[1][2]
    yb = out[1][3]

    grids = {}

    for grid_name, res_x, res_y in zip(names, degreesx, degreesy):
        factor_x = res_x / ref_degree_x
        factor_y = res_y / ref_degree_y
        grid_tile_size_x = tile_size_x * factor_x
        grid_tile_size_y = tile_size_y * factor_y
        tiles, features = _create_tiles(tile_size=(grid_tile_size_x, grid_tile_size_y),
                                        grid_x_min=x_min, grid_y_max=y_max,
                                        bbox=(xl, xr, yb, yu),
                                        srid=srid)
        grids[grid_name] = dict(
            tiles=tiles,
            features=features
        )

    return grids, tile_srs_p4
