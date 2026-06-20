"""
Phase 1: Data Pipeline for Road Extraction
Maps to:
  - 4 Core Objectives:
    * Occlusion-Aware Extraction (Data preparation & augmentation)
  - Evaluation Parameters:
    * IoU & Dice Score (Training targets generation)
    * Generalisation (Diverse AOI training data)
"""

import logging
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio import features as rio_features
from pyproj import CRS
from shapely.geometry import mapping, LineString, MultiLineString

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default UTM zone for Bengaluru AOI — EPSG:32643 (UTM zone 43N)
DEFAULT_UTM_EPSG = 32643

# Road buffer widths in meters, keyed by OSM 'highway' tag value
ROAD_BUFFER_M: Dict[str, float] = {
    'motorway': 10.0, 'motorway_link': 8.0,
    'trunk': 10.0, 'trunk_link': 8.0,
    'primary': 10.0, 'primary_link': 8.0,
    'secondary': 8.0, 'secondary_link': 7.0,
    'tertiary': 8.0, 'tertiary_link': 7.0,
    'residential': 6.0, 'unclassified': 6.0,
    'service': 5.0, 'living_street': 5.0,
    'pedestrian': 4.0, 'track': 4.0, 'path': 3.0,
}
DEFAULT_BUFFER_M = 6.0

# Fixed seed for reproducible hard-negative sampling
_NEG_SAMPLE_SEED = 42
_NEG_KEEP_RATIO = 0.20  # keep 20% of empty tiles


def build_binary_road_mask(
    mask_rgb: np.ndarray,
    class_dict_df: "pd.DataFrame",
    road_class_name: str = "road",
    tolerance: int = 5,
) -> np.ndarray:
    """
    Convert a DeepGlobe-style RGB class mask to a binary road mask.

    Args:
        mask_rgb: (H, W, 3) uint8 RGB mask where pixel colours encode classes.
        class_dict_df: DataFrame from class_dict.csv with columns for class name
                       and its R, G, B values.
        road_class_name: Name of the road class (case-insensitive match).
        tolerance: Per-channel tolerance for colour matching (handles JPEG
                   compression artifacts if masks aren't lossless PNG).

    Returns:
        (H, W) uint8 binary mask: 1 = road, 0 = background.
    """
    import pandas as pd

    # Find the name column (may be 'name', 'class', 'class_name', etc.)
    name_col = None
    for col in class_dict_df.columns:
        if col.strip().lower() in ("name", "class", "class_name", "label"):
            name_col = col
            break
    if name_col is None:
        # Fallback: assume first non-numeric column
        name_col = class_dict_df.select_dtypes(include="object").columns[0]

    # Find R, G, B columns
    col_map = {}
    for col in class_dict_df.columns:
        cl = col.strip().lower()
        if cl in ("r", "red"):
            col_map["r"] = col
        elif cl in ("g", "green"):
            col_map["g"] = col
        elif cl in ("b", "blue"):
            col_map["b"] = col

    # Locate the road class row (case-insensitive)
    road_row = class_dict_df[
        class_dict_df[name_col].str.strip().str.lower() == road_class_name.lower()
    ]
    if road_row.empty:
        raise ValueError(
            f"Road class '{road_class_name}' not found in class_dict. "
            f"Available classes: {class_dict_df[name_col].tolist()}"
        )

    road_r = int(road_row.iloc[0][col_map["r"]])
    road_g = int(road_row.iloc[0][col_map["g"]])
    road_b = int(road_row.iloc[0][col_map["b"]])

    # Match with tolerance to handle JPEG compression artifacts
    mask_f = mask_rgb.astype(np.int16)
    match = (
        (np.abs(mask_f[:, :, 0] - road_r) <= tolerance) &
        (np.abs(mask_f[:, :, 1] - road_g) <= tolerance) &
        (np.abs(mask_f[:, :, 2] - road_b) <= tolerance)
    )

    return match.astype(np.uint8)


def fetch_osm_roads(
    aoi_bbox: Tuple[float, float, float, float],
    target_epsg: int = DEFAULT_UTM_EPSG,
) -> gpd.GeoDataFrame:
    """Fetches OSM road vectors for an AOI and reprojects to a projected CRS (meters)."""
    north, south, east, west = aoi_bbox

    # Fetch all highway features within the bounding box
    gdf = ox.features_from_bbox(
        bbox=(north, south, east, west),
        tags={'highway': True},
    )

    # Keep only line geometries (roads); drop points (traffic signals etc.) and polygons (areas)
    gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])].copy()

    if gdf.empty:
        logger.warning("No road geometries found in the AOI. Returning empty GeoDataFrame.")
        return gdf

    # OSMnx returns EPSG:4326 — reproject to projected UTM immediately
    gdf = gdf.to_crs(epsg=target_epsg)

    # Normalise the highway column (it can be a list for multi-tagged features)
    if 'highway' in gdf.columns:
        gdf['highway'] = gdf['highway'].apply(
            lambda v: v[0] if isinstance(v, list) else v
        )

    logger.info(f"Fetched {len(gdf)} road segments, reprojected to EPSG:{target_epsg}")
    return gdf


def rasterize_osm_to_mask(
    gdf: gpd.GeoDataFrame,
    raster_profile: Dict[str, Any],
) -> np.ndarray:
    """Rasterizes buffered road vectors into a binary mask aligned to a satellite raster profile."""
    transform = raster_profile['transform']
    height = raster_profile['height']
    width = raster_profile['width']

    # Ensure the GeoDataFrame CRS matches the raster CRS
    raster_crs = raster_profile.get('crs')
    if raster_crs is not None and gdf.crs != CRS.from_user_input(raster_crs):
        gdf = gdf.to_crs(raster_crs)

    # Buffer each road geometry by its highway-class width
    def _buffer_geom(row):
        hw = row.get('highway', '')
        buf = ROAD_BUFFER_M.get(str(hw), DEFAULT_BUFFER_M)
        return row.geometry.buffer(buf)

    buffered = gdf.apply(_buffer_geom, axis=1)

    # Build (geometry, value) pairs for rasterio rasterize
    shapes = [(mapping(geom), 1) for geom in buffered if geom is not None and not geom.is_empty]

    if not shapes:
        logger.warning("No valid buffered geometries to rasterize — returning empty mask.")
        return np.zeros((height, width), dtype=np.uint8)

    mask = rio_features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    road_pct = mask.sum() / mask.size * 100
    logger.info(f"Rasterized mask: {width}x{height}, road coverage {road_pct:.2f}%")
    return mask


def tile_image_and_mask(
    image_path: str,
    mask_array: np.ndarray,
    tile_size: int = 512,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Tiles a satellite image and its aligned mask into patches with hard-negative mining."""
    rng = np.random.default_rng(_NEG_SAMPLE_SEED)
    tiles: List[Tuple[str, np.ndarray, np.ndarray]] = []

    with rasterio.open(image_path) as src:
        # Read all bands -> shape (C, H, W)
        img = src.read()
        # Transpose to (H, W, C) for tiling convenience
        img = np.transpose(img, (1, 2, 0))

    h, w = img.shape[:2]
    mh, mw = mask_array.shape[:2]

    if h != mh or w != mw:
        raise ValueError(
            f"Image ({h}x{w}) and mask ({mh}x{mw}) dimensions do not match. "
            "Ensure they share the same raster profile / CRS."
        )

    for row_idx in range(0, h - tile_size + 1, tile_size):
        for col_idx in range(0, w - tile_size + 1, tile_size):
            img_tile = img[row_idx:row_idx + tile_size, col_idx:col_idx + tile_size]
            msk_tile = mask_array[row_idx:row_idx + tile_size, col_idx:col_idx + tile_size]

            has_road = msk_tile.sum() > 0

            if has_road:
                # Always keep tiles containing road pixels
                tiles.append((f"{row_idx}_{col_idx}", img_tile, msk_tile))
            else:
                # Hard-negative mining: keep only ~20% of empty tiles
                if rng.random() < _NEG_KEEP_RATIO:
                    tiles.append((f"{row_idx}_{col_idx}", img_tile, msk_tile))

    logger.info(f"Tiled {len(tiles)} patches (tile_size={tile_size}) from {image_path}")
    return tiles


def add_synthetic_occlusion(
    image: np.ndarray,
    mask: np.ndarray,
    occlusion_type: str = "shadow",
) -> np.ndarray:
    """Applies synthetic occlusion (shadow/canopy/cloud) to an image tile without altering the mask."""
    h, w = image.shape[:2]
    occluded = image.astype(np.float32).copy()

    # --- Generate an irregular soft-edged blob mask ---
    blob = np.zeros((h, w), dtype=np.float32)

    # Place 3-7 overlapping random ellipses for an organic shape
    rng = np.random.default_rng()  # non-deterministic per call for augmentation variety
    n_ellipses = rng.integers(3, 8)
    # Target coverage: 10-35% of tile area
    target_area_fraction = rng.uniform(0.10, 0.35)
    avg_radius = int(np.sqrt(target_area_fraction * h * w / (np.pi * n_ellipses)))
    avg_radius = max(avg_radius, 10)  # floor to prevent degenerate tiny blobs

    for _ in range(n_ellipses):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        ax1 = rng.integers(int(avg_radius * 0.5), int(avg_radius * 1.5) + 1)
        ax2 = rng.integers(int(avg_radius * 0.5), int(avg_radius * 1.5) + 1)
        angle = rng.integers(0, 180)
        cv2.ellipse(blob, (int(cx), int(cy)), (int(ax1), int(ax2)),
                    int(angle), 0, 360, 1.0, -1)

    # Soft-edge the blob boundary with Gaussian blur
    ksize = max(15, (avg_radius // 2) * 2 + 1)  # must be odd
    blob = cv2.GaussianBlur(blob, (ksize, ksize), 0)
    # Renormalize to [0, 1] after blur
    blob_max = blob.max()
    if blob_max > 0:
        blob = blob / blob_max

    # --- Apply type-specific occlusion ---
    if occlusion_type == "shadow":
        # Darken pixels under the blob (factor 0.3–0.5)
        darkness = rng.uniform(0.3, 0.5)
        for c in range(occluded.shape[2]):
            occluded[:, :, c] = occluded[:, :, c] * (1.0 - blob * (1.0 - darkness))

    elif occlusion_type == "canopy":
        # Blend toward a green/dark-green noisy texture
        green_base = np.zeros_like(occluded)
        green_base[:, :, 1] = rng.uniform(60, 120)  # green channel
        green_base[:, :, 0] = rng.uniform(20, 50)    # blue
        green_base[:, :, 2] = rng.uniform(20, 60)    # red
        # Add per-pixel noise for texture
        noise = rng.normal(0, 15, size=green_base.shape).astype(np.float32)
        green_base = np.clip(green_base + noise, 0, 255)

        alpha = blob[:, :, np.newaxis] * rng.uniform(0.5, 0.85)
        occluded = occluded * (1.0 - alpha) + green_base * alpha

    elif occlusion_type == "cloud":
        # Blend toward a bright white/light-gray semi-opaque overlay
        cloud_val = rng.uniform(200, 255)
        cloud_layer = np.full_like(occluded, cloud_val)
        # Add subtle noise for realism
        noise = rng.normal(0, 10, size=cloud_layer.shape).astype(np.float32)
        cloud_layer = np.clip(cloud_layer + noise, 0, 255)

        alpha = blob[:, :, np.newaxis] * rng.uniform(0.4, 0.75)
        occluded = occluded * (1.0 - alpha) + cloud_layer * alpha

    else:
        raise ValueError(f"Unknown occlusion_type '{occlusion_type}'. Use 'shadow', 'canopy', or 'cloud'.")

    return np.clip(occluded, 0, 255).astype(np.uint8)
