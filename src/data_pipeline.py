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
from typing import Tuple, List, Dict, Any, Union

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
    """Fetches OSM road vectors for an AOI and reprojects to a projected CRS (meters), using local cache if available."""
    import hashlib
    import os

    # 1. Determine cache path based on hashed bbox (rounded to 6 decimal places)
    north, south, east, west = aoi_bbox
    rounded_bbox = [round(c, 6) for c in aoi_bbox]
    bbox_str = f"{rounded_bbox[0]:.6f}_{rounded_bbox[1]:.6f}_{rounded_bbox[2]:.6f}_{rounded_bbox[3]:.6f}"
    bbox_hash = hashlib.sha256(bbox_str.encode('utf-8')).hexdigest()
    
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "osm_cache")
    cache_file = os.path.join(cache_dir, f"{bbox_hash}.geojson")

    # 2. Check cache first
    if os.path.exists(cache_file):
        try:
            gdf = gpd.read_file(cache_file)
            if not gdf.empty:
                logger.info(f"[OSM CACHE HIT] Loaded cached OSM data from outputs/osm_cache/{bbox_hash}.geojson")
                # Reproject if loaded CRS doesn't match target_epsg
                if gdf.crs is None or gdf.crs.to_epsg() != target_epsg:
                    gdf = gdf.to_crs(epsg=target_epsg)
                return gdf
            else:
                logger.info(f"[OSM CACHE HIT] Cached file outputs/osm_cache/{bbox_hash}.geojson is empty, returning empty GeoDataFrame.")
                return gpd.GeoDataFrame(geometry=[], crs=target_epsg)
        except Exception as e:
            logger.warning(f"Error loading OSM cache from {cache_file}: {e}. Falling back to live query.")

    # 3. Attempt Live Fetch
    logger.info(f"[OSM CACHE MISS] Attempting live OSM query for bbox {aoi_bbox}")
    try:
        # Fetch all highway features within the bounding box
        # OSMnx expects bbox as (left, bottom, right, top) -> (west, south, east, north)
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={'highway': True},
        )
        
        # Keep only line geometries (roads); drop points and polygons
        gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
        
        if gdf.empty:
            logger.warning("No road geometries found in live OSM query. Saving empty cache.")
            os.makedirs(cache_dir, exist_ok=True)
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            empty_gdf.to_file(cache_file, driver="GeoJSON")
            return gpd.GeoDataFrame(geometry=[], crs=target_epsg)

        # OSMnx returns EPSG:4326 — reproject to projected UTM immediately
        gdf = gdf.to_crs(epsg=target_epsg)

        # Normalise the highway column (it can be a list for multi-tagged features)
        if 'highway' in gdf.columns:
            gdf['highway'] = gdf['highway'].apply(
                lambda v: v[0] if isinstance(v, list) else v
            )

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        try:
            # We can save it in projected CRS, geopandas handles it
            gdf.to_file(cache_file, driver="GeoJSON")
            logger.info(f"[OSM CACHE WRITE] Successfully cached live OSM query to outputs/osm_cache/{bbox_hash}.geojson")
        except Exception as save_err:
            logger.warning(f"Failed to save OSM query to cache: {save_err}")

        logger.info(f"Fetched {len(gdf)} road segments from live network, reprojected to EPSG:{target_epsg}")
        return gdf

    except Exception as live_err:
        # If live query fails, check if cache file exists as fallback
        if os.path.exists(cache_file):
            logger.info(f"[OSM FALLBACK] Live query failed ({live_err}). Recovering from cache outputs/osm_cache/{bbox_hash}.geojson.")
            try:
                gdf = gpd.read_file(cache_file)
                if gdf.crs is None or gdf.crs.to_epsg() != target_epsg:
                    gdf = gdf.to_crs(epsg=target_epsg)
                return gdf
            except Exception as read_err:
                logger.error(f"Fallback cache load failed: {read_err}")
                
        logger.warning(f"Live OSM query failed and no cache is available. Returning empty GeoDataFrame. Error: {live_err}")
        return gpd.GeoDataFrame(geometry=[], crs=target_epsg)


def validate_satellite_image(img_path: str) -> Tuple[bool, str]:
    """
    Validates that the image at img_path is a natural satellite image and not a presentation slide,
    screenshot, or UI frame.
    
    Returns:
        (is_valid, reason_or_status)
    """
    import cv2
    import numpy as np
    import os
    import rasterio

    # Check file extension
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in ('.tif', '.tiff', '.jpeg', '.jpg', '.png'):
        return False, f"Invalid file extension: {ext}"

    # Read image dimensions using rasterio or cv2
    try:
        if ext in ('.tif', '.tiff'):
            with rasterio.open(img_path) as src:
                w, h = src.width, src.height
                has_crs = src.crs is not None
                num_channels = src.count
        else:
            img = cv2.imread(img_path)
            if img is None:
                return False, "Failed to load image via OpenCV"
            h, w, num_channels = img.shape
            has_crs = False
    except Exception as e:
        return False, f"Read error: {e}"

    # Rule 1: Aspect Ratio Check
    # Satellite imagery patches/tiles are square or near-square (typically 1:1).
    # Slides are widescreen (16:9 = 1.77, 4:3 = 1.33).
    aspect_ratio = w / h
    if aspect_ratio > 1.3 or aspect_ratio < 0.7:
        if not has_crs:
            return False, f"Non-square aspect ratio ({aspect_ratio:.2f}), likely a presentation slide or widescreen screenshot"

    # Load full image for visual audits
    try:
        if ext in ('.tif', '.tiff'):
            with rasterio.open(img_path) as src:
                bands = min(3, src.count)
                img = src.read(list(range(1, bands + 1)))
                img = np.transpose(img, (1, 2, 0))
                if bands == 1:
                    img = np.repeat(img, 3, axis=2)
                num_channels = 3
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return False, f"Visual load error: {e}"

    # Rule 2: Color distribution / flat regions (Slides backgrounds)
    # Quantize colors to 3 bits per channel (8 bins per channel -> 512 total colors)
    quantized = (img >> 5).astype(np.int32)
    flat_colors = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
    unique_bins, counts = np.unique(flat_colors, return_counts=True)
    dominant_fraction = np.max(counts) / (h * w)

    # Rule 3: High contrast edges (Laplacian Variance)
    # Slides with text characters have extremely high contrast local changes.
    # Satellite images have textured but smooth landscapes.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if num_channels >= 3 else img[:, :, 0]
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Rule 4: Edge density
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_density = np.mean(np.abs(dx) > 20)

    # Reject if it doesn't have GeoTIFF CRS metadata and exhibits slide/UI characteristics
    if not has_crs:
        # Check for square presentation slides or diagrams
        # Usually they have high flat-color dominance, very few unique quantized colors, and high contrast text
        if dominant_fraction > 0.50 and len(unique_bins) < 100 and laplacian_var > 1000.0:
            return False, f"High flat-color dominance ({dominant_fraction:.2f}) with restricted palette ({len(unique_bins)} bins) and high contrast text (Laplacian Var = {laplacian_var:.1f}), likely a diagram or slide"
        
        # Check for screenshots with UI components or text documents
        # (mostly solid backgrounds with sharp high-contrast text and thin lines)
        if dominant_fraction > 0.70 and edge_density > 0.15:
            return False, f"High flat-color dominance ({dominant_fraction:.2f}) and high edge density ({edge_density:.4f}), typical of UI screenshots"

    return True, "Valid satellite image"


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
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Applies synthetic occlusion (shadow/canopy/cloud) to an image tile without altering the mask."""
    from typing import Union
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

    result = np.clip(occluded, 0, 255).astype(np.uint8)
    if return_mask:
        return result, blob
    return result


def classify_terrain_tile(image: np.ndarray) -> str:
    """
    Classifies an RGB tile into 'urban', 'forested', or 'rural' using color
    and texture heuristics.
    """
    import cv2
    img = image.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    
    # NDVI-like proxy: Green-Red Ratio (GRR)
    grr = (g - r) / (g + r + 1e-8)
    mean_grr = np.mean(grr)
    
    # Texture proxy: variance of Laplacian gradients
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = laplacian.var()
    
    # Classification Logic
    if mean_grr > 0.08:
        return "forested"
    elif texture_var > 250.0:
        return "urban"
    else:
        return "rural"


def run_stratified_evaluation(
    images: List[np.ndarray],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    unhealed_graphs: List["nx.Graph"],
    healed_graphs: List["nx.Graph"],
) -> Dict[str, Dict[str, float]]:
    """
    Groups evaluation metrics by terrain category and returns a comparison dictionary.
    """
    from segmentation_train import compute_metrics
    from centrality import compute_connectivity_ratio
    import networkx as nx
    from typing import Dict
    
    results = {
        "urban": {"iou": [], "dice": [], "conn_ratio": []},
        "forested": {"iou": [], "dice": [], "conn_ratio": []},
        "rural": {"iou": [], "dice": [], "conn_ratio": []}
    }
    
    for img, pred, gt, g_un, g_he in zip(images, pred_masks, gt_masks, unhealed_graphs, healed_graphs):
        terrain = classify_terrain_tile(img)
        
        # Convert pred and gt to PyTorch tensors for compute_metrics compatibility
        import torch
        # Scale to simulate model logit output: 0 -> -5.0, 1 -> 5.0
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float() * 10.0 - 5.0
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
        
        metrics = compute_metrics(pred_t, gt_t)
        conn = compute_connectivity_ratio(g_un, g_he)
        
        results[terrain]["iou"].append(metrics["iou"])
        results[terrain]["dice"].append(metrics["dice"])
        results[terrain]["conn_ratio"].append(conn)
        
    # Aggregate mean values
    report = {}
    for terrain, metrics_list in results.items():
        if metrics_list["iou"]:
            report[terrain] = {
                "mean_iou": float(np.mean(metrics_list["iou"])),
                "mean_dice": float(np.mean(metrics_list["dice"])),
                "mean_conn_ratio": float(np.mean(metrics_list["conn_ratio"])),
                "sample_count": len(metrics_list["iou"])
            }
        else:
            report[terrain] = {
                "mean_iou": 0.0,
                "mean_dice": 0.0,
                "mean_conn_ratio": 1.0,
                "sample_count": 0
            }
            
    return report
