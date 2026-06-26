import os
import sys
import glob
import pickle
import json
import numpy as np
from PIL import Image
import affine
import networkx as nx
import torch
import argparse
import cv2
import warnings

# ---------------------------------------------------------------------------
# Allow importing src/ modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from segmentation_train import build_model, predict_mask
from skeleton_to_graph import mask_to_graph
from graph_healing import heal_graph_connectivity
from spatial_stitching import spatial_stitch_nodes
from centrality import compute_betweenness_centrality, export_criticality_geojson

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

def load_and_preprocess_raster(
    img_path: str,
    target_crs_epsg: int = 32643,
    target_resolution: float = 1.0,
    tile_size: int = 512
) -> tuple[np.ndarray, affine.Affine, int]:
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

def main():
    print("="*60)
    print("ROUTE RESILIENCE BATCH PIPELINE RUNNER")
    print("="*60)
    
    # Argparse configuration
    parser = argparse.ArgumentParser(description="Route Resilience Batch Pipeline Runner")
    parser.add_argument("--fast-demo", action="store_true", help="Use cached results to speed up live demo")
    parser.add_argument("--threshold", type=float, default=0.5, help="Road mask prediction confidence threshold")
    args = parser.parse_args()
    
    # 1. Paths
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    unhealed_pkl_path = os.path.join(outputs_dir, "unhealed_graph.pkl")
    healed_pkl_path = os.path.join(outputs_dir, "healed_graph.pkl")
    centrality_json_path = os.path.join(outputs_dir, "centrality.json")
    geojson_path = os.path.join(outputs_dir, "criticality.geojson")

    # Fast-demo mode cache check
    if args.fast_demo:
        if (os.path.exists(unhealed_pkl_path) and
            os.path.exists(healed_pkl_path) and
            os.path.exists(centrality_json_path) and
            os.path.exists(geojson_path)):
            print("[INFO] Fast-demo mode active: Loaded cached graphs and centrality from outputs/.")
            print("OK: PIPELINE BATCH PROCESS COMPLETED (CACHED)!")
            return
        else:
            print("[WARN] Fast-demo requested but cached files are missing. Running full pipeline.")

    # Prefer real satellite images from local_validation; fall back to hacthon_req
    real_sat_dir = os.path.join(PROJECT_ROOT, "data", "local_validation", "images")
    real_images = sorted(glob.glob(os.path.join(real_sat_dir, "*.png")) +
                         glob.glob(os.path.join(real_sat_dir, "*.jpg")) +
                         glob.glob(os.path.join(real_sat_dir, "*.tif*")))

    if real_images:
        image_files = real_images
        print(f"Found {len(image_files)} real satellite images in data/local_validation/images/.")
    else:
        # Fallback: hackathon request folder
        fallback_pattern = os.path.join(PROJECT_ROOT, "hacthon_req", "*.jpeg")
        image_files = sorted(glob.glob(fallback_pattern))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "hacthon_req", "*.tif*")))
        if not image_files:
            print(f"[ERROR] No images found in {real_sat_dir} or hacthon_req/")
            return
        print(f"Found {len(image_files)} hackathon images to process (fallback).")
    
    # 2. Check model weights
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Error: Model weights not found at {checkpoint_path}")
        print("Please ensure best_model.pth is in the checkpoints/ folder.")
        return
        
    # 3. Load Model
    print("\n1. Loading trained segmentation model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    try:
        # Avoid OpenMP issues
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        state_dict = torch.load(checkpoint_path, map_location=device)
        has_attention = any(k.startswith("base_model.") for k in state_dict.keys())
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint file: {e}")
        return
        
    model = build_model(
        architecture="Unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        attention=has_attention
    )
    
    try:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"   Model loaded successfully (attention={has_attention}).")
    except Exception as e:
        print(f"[ERROR] Error loading model state dict: {e}")
        return

    # 4. Batch Process loop
    combined_unhealed_graph = nx.Graph()
    tile_summaries = []
    
    METERS_PER_PIXEL = 1.0 
    TILE_SIZE = 512
    TILE_WIDTH_METERS = TILE_SIZE * METERS_PER_PIXEL
    TILE_HEIGHT_METERS = TILE_SIZE * METERS_PER_PIXEL
    
    print("\n2. Processing individual image tiles and compiling mosaic graph...")
    
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        
        # Load and preprocess using rasterio
        try:
            img_np, raster_transform, src_crs_epsg = load_and_preprocess_raster(
                img_path, target_crs_epsg=32643, target_resolution=1.0, tile_size=TILE_SIZE
            )
        except Exception as e:
            print(f"   [ERROR] Error reading/reprojecting {img_name}: {e}")
            tile_summaries.append({
                "name": img_name, "road_pixels": 0, "nodes": 0, "edges": 0, "status": "ReadError"
            })
            continue

        # Model Inference
        road_mask = predict_mask(model, img_np, device=device, threshold=args.threshold)
        road_pixels = int(road_mask.sum())
        
        # Graceful empty mask handling per tile
        if road_pixels < 50:
            # Try a lower threshold to capture faint features
            road_mask_fallback = predict_mask(model, img_np, device=device, threshold=max(0.2, args.threshold - 0.2))
            road_pixels_fallback = int(road_mask_fallback.sum())
            if road_pixels_fallback >= 50:
                print(f"   [INFO]  {img_name}: Low confidence detection used (pixels: {road_pixels_fallback})")
                road_mask = road_mask_fallback
                road_pixels = road_pixels_fallback
            else:
                print(f"   [WARN]  {img_name}: No significant road detected ({road_pixels} px), skipping.")
                tile_summaries.append({
                    "name": img_name, "road_pixels": road_pixels, "nodes": 0, "edges": 0, "status": "Skipped"
                })
                continue
            
        # Handle coordinate system transform setup
        source_crs_epsg = 32643
        if src_crs_epsg == -1:
            # JPEG fallback: apply mock grid offset
            cols = 4
            grid_col = idx % cols
            grid_row = idx // cols
            x_offset = grid_col * TILE_WIDTH_METERS
            y_offset = grid_row * (-TILE_HEIGHT_METERS)
            base_x = 600000.0
            base_y = 1430000.0
            
            # Construct specific transform with offset
            tile_transform = affine.Affine(
                METERS_PER_PIXEL, 0.0, base_x + x_offset,
                0.0, -METERS_PER_PIXEL, base_y + y_offset
            )
        else:
            tile_transform = raster_transform
            source_crs_epsg = src_crs_epsg
        
        # Convert road mask to georeferenced graph
        try:
            tile_graph = mask_to_graph(
                binary_mask=road_mask,
                raster_transform=tile_transform,
                source_crs_epsg=source_crs_epsg,
                target_utm_epsg=32643,
                min_branch_length_px=5
            )
        except Exception as e:
            print(f"   [ERROR] Error extracting graph for {img_name}: {e}")
            tile_summaries.append({
                "name": img_name, "road_pixels": road_pixels, "nodes": 0, "edges": 0, "status": "GraphError"
            })
            continue
            
        n_nodes = tile_graph.number_of_nodes()
        n_edges = tile_graph.number_of_edges()
        
        if n_nodes == 0:
            print(f"   [WARN]  {img_name}: Extracted graph is empty, skipping.")
            tile_summaries.append({
                "name": img_name, "road_pixels": road_pixels, "nodes": 0, "edges": 0, "status": "EmptyGraph"
            })
            continue
            
        # Relabel nodes to prevent collisions
        relabel_map = {node: f"tile_{idx}_node_{node}" for node in tile_graph.nodes}
        tile_graph = nx.relabel_nodes(tile_graph, relabel_map)
        
        combined_unhealed_graph = nx.compose(combined_unhealed_graph, tile_graph)
        print(f"   [OK] {img_name}: Extracted {n_nodes} nodes, {n_edges} edges. Merged.")
        
        tile_summaries.append({
            "name": img_name, "road_pixels": road_pixels, "nodes": n_nodes, "edges": n_edges, "status": "Merged"
        })
        
    # Graceful handling if combined graph is empty
    if combined_unhealed_graph.number_of_nodes() == 0:
        print("[WARNING] Combined graph is empty. Creating a dummy single-node network to prevent crashes.")
        combined_unhealed_graph.add_node("dummy_node", x=600000.0, y=1430000.0)
        
    # 5. Spatial Boundary Stitching
    print("\n3. Running spatial stitching on tile boundaries...")
    stitched_unhealed_graph = spatial_stitch_nodes(combined_unhealed_graph, tolerance_m=2.0)
    
    # 6. Graph Healing on the Stitched Graph
    print("\n4. Running topological graph healing on combined network...")
    healed_graph = heal_graph_connectivity(
        stitched_unhealed_graph,
        max_bridge_distance_m=100.0,
        angular_tolerance_deg=30.0
    )
    
    # 7. Centrality Analysis
    print("\n5. Analyzing centrality/criticality scores...")
    centrality_scores = compute_betweenness_centrality(healed_graph)
    
    # 8. Export Outputs
    print("\n6. Exporting processed results to /outputs/...")
    
    with open(unhealed_pkl_path, "wb") as f:
        pickle.dump(combined_unhealed_graph, f)
    with open(healed_pkl_path, "wb") as f:
        pickle.dump(healed_graph, f)
        
    centrality_json_dict = {str(k): float(v) for k, v in centrality_scores.items()}
    with open(centrality_json_path, "w") as f:
        json.dump(centrality_json_dict, f, indent=2)
        
    export_criticality_geojson(healed_graph, centrality_scores, geojson_path, source_crs_epsg=32643)
    
    def _lcc_size(g):
        if len(g) == 0:
            return 0
        return max(len(c) for c in nx.connected_components(g))
        
    lcc_before = _lcc_size(combined_unhealed_graph)
    lcc_after = _lcc_size(healed_graph)
    
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"{'Image Name':<15} | {'Road Pixels':<12} | {'Nodes':<6} | {'Edges':<6} | {'Status':<8}")
    print("-"*60)
    for s in tile_summaries:
        print(f"{s['name']:<15} | {s['road_pixels']:<12} | {s['nodes']:<6} | {s['edges']:<6} | {s['status']:<8}")
    print("="*60)
    print("COMBINED GRAPH STATS:")
    print(f"  - Total Nodes               : {combined_unhealed_graph.number_of_nodes()}")
    print(f"  - Total Edges (Before Heal) : {combined_unhealed_graph.number_of_edges()}")
    print(f"  - Total Edges (After Heal)  : {healed_graph.number_of_edges()}")
    print(f"  - LCC Size (Before Heal)    : {lcc_before}")
    print(f"  - LCC Size (After Heal)     : {lcc_after}")
    print("="*60)
    print("OK: PIPELINE BATCH PROCESS COMPLETED!")
    print("Open or refresh the Dashboard (http://localhost:8501) to see the results!")

if __name__ == "__main__":
    main()
