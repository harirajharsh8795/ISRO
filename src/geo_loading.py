import os
import cv2
import numpy as np
import warnings
import affine
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from typing import Tuple

def load_and_preprocess_raster(
    img_path: str,
    target_crs_epsg: int = 32643,
    target_resolution: float = 1.0,
    tile_size: int = 512
) -> Tuple[np.ndarray, affine.Affine, int]:
    """
    Loads an image using rasterio, extracts CRS/transform, reprojects to the target
    projected CRS at the target resolution, and returns an RGB numpy array, 
    the updated affine transform, and the source CRS EPSG.
    """
    with rasterio.open(img_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        
        # Check if CRS is missing
        if not src_crs:
            warnings.warn(
                f"[WARNING] {os.path.basename(img_path)} lacks valid CRS metadata! "
                "Falling back to local pixel coordinates."
            )
            # Fallback local transform (1m per pixel, origin at 0,0)
            src_crs = CRS.from_epsg(target_crs_epsg)
            src_transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
            source_crs_epsg = target_crs_epsg
            is_fake = True
        else:
            source_crs_epsg = src_crs.to_epsg()
            if source_crs_epsg is None:
                source_crs_epsg = target_crs_epsg
            is_fake = False

        # Determine target transform and dimensions
        target_crs = CRS.from_epsg(target_crs_epsg)
        
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height,
            left=src.bounds.left, bottom=src.bounds.bottom,
            right=src.bounds.right, top=src.bounds.top,
            resolution=target_resolution
        )
        
        # Read the bands (assume first 3 bands for RGB)
        num_bands = min(3, src.count)
        bands_data = src.read(range(1, num_bands + 1))
        
        if num_bands == 1:
            bands_data = np.repeat(bands_data, 3, axis=0)
        elif num_bands == 2:
            bands_data = np.concatenate([bands_data, bands_data[:1]], axis=0)

        # Allocate output array
        dst_data = np.zeros((3, dst_height, dst_width), dtype=bands_data.dtype)

        # Reproject
        for b in range(3):
            reproject(
                source=bands_data[b],
                destination=dst_data[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
        img_np = np.transpose(dst_data, (1, 2, 0))
        
        # Resize to standard tile size for model inference, updating transform
        h, w = img_np.shape[:2]
        if h != tile_size or w != tile_size:
            scale_x = w / tile_size
            scale_y = h / tile_size
            resized_transform = dst_transform * rasterio.Affine.scale(scale_x, scale_y)
            img_np = cv2.resize(img_np, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        else:
            resized_transform = dst_transform

        if img_np.max() <= 1.0 and img_np.dtype == np.float32:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Flag fake transform so caller knows to apply grid offset if mock
        if is_fake:
            return img_np, resized_transform, -1
        return img_np, resized_transform, source_crs_epsg

def pixel_to_geo(r: float, c: float, transform: affine.Affine) -> Tuple[float, float]:
    """
    Converts pixel row/col coordinates to projected geographic coordinates (x, y)
    using the provided affine transform matrix.
    """
    # rasterio convention: x, y = transform * (col, row)
    x, y = transform * (c, r)
    return x, y
