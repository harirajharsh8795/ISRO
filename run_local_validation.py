import os
import sys
import glob
import json
import torch
import cv2
import numpy as np
import networkx as nx
import argparse
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_pipeline import validate_satellite_image, fetch_osm_roads
from segmentation_train import build_model, predict_mask, get_val_augmentations
from skeleton_to_graph import mask_to_graph
from graph_healing import heal_graph_connectivity
from spatial_stitching import spatial_stitch_nodes
from centrality import compute_betweenness_centrality, export_criticality_geojson
from geo_loading import load_and_preprocess_raster

def print_setup_guide():
    print("=" * 60)
    print("LOCAL VALIDATION DATASET SETUP GUIDE")
    print("=" * 60)
    print("\nNo local validation imagery found in data/local_validation/images/")
    print("Please follow the instructions below to obtain and set up real satellite data:\n")
    print("Step 1: Install Kaggle CLI:")
    print("  pip install kaggle\n")
    print("Step 2: Download DeepGlobe Road Extraction Dataset:")
    print("  kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset")
    print("  Unzip it and copy 5-10 images to: data/local_validation/images/")
    print("  Copy corresponding mask PNGs to: data/local_validation/masks/\n")
    print("Step 3: Download HybridSAR Road Dataset (HSRD):")
    print("  kaggle datasets download -d satyveeryadav/the-hybridsar-road-dataset-hsrd")
    print("  Unzip and place sample imagery in: data/local_validation/images/\n")
    print("After placing the files, run this command again to execute validation:")
    print("  python run_local_validation.py\n")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Route Resilience Local Validation Setup")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction confidence threshold")
    args = parser.parse_args()

    image_dir = os.path.join(PROJECT_ROOT, "data", "local_validation", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "local_validation", "masks")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs", "local_validation")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Check if dataset directory is empty
    image_pattern = os.path.join(image_dir, "*.*")
    all_image_paths = glob.glob(image_pattern)
    
    # Remove directories from list
    all_image_paths = [p for p in all_image_paths if os.path.isfile(p)]
    
    if not all_image_paths:
        print_setup_guide()
        return

    print("=" * 60)
    print("LOCAL VALIDATION: DATASET INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Scanning directory: {image_dir}")
    print(f"Found {len(all_image_paths)} total files. Running validation filters...\n")

    manifest = []
    valid_images = []
    
    # 2. Audit and Validate image-mask pairs
    for img_path in all_image_paths:
        img_name = os.path.basename(img_path)
        is_valid, reason = validate_satellite_image(img_path)
        
        # Check matching mask status
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        has_mask = os.path.exists(mask_path)
        
        status = "Accepted" if is_valid else "Rejected"
        print(f"  - {img_name:<20} | Status: {status:<10} | Reason: {reason}")
        
        with Image.open(img_path) as im:
            width, height = im.size
            
        manifest.append({
            "filename": img_name,
            "resolution": f"{width}x{height}",
            "source_folder": os.path.relpath(image_dir, PROJECT_ROOT),
            "status": status,
            "reason": reason,
            "has_matching_mask": has_mask
        })
        
        if is_valid:
            valid_images.append((img_path, mask_path if has_mask else None))

    # Write manifest
    manifest_path = os.path.join(outputs_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWritten dataset manifest to: outputs/local_validation/dataset_manifest.json")

    print(f"\nAudit complete: {len(valid_images)} accepted, {len(all_image_paths) - len(valid_images)} rejected.")
    
    if not valid_images:
        print("[ERROR] No valid satellite images survived the validation filters. Aborting pipeline.")
        return

    # 3. Load Model
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Model weights not found at {checkpoint_path}")
        return

    print("\n1. Loading trained segmentation model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    has_attention = any(k.startswith("base_model.") for k in state_dict.keys())
    
    model = build_model(
        architecture="Unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        attention=has_attention
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("   Model loaded successfully.")

    # 4. Process Valid Images
    combined_unhealed_graph = nx.Graph()
    tile_summaries = []
    
    print("\n2. Running inference and graph compilation on valid images...")
    
    for idx, (img_path, mask_path) in enumerate(valid_images):
        img_name = os.path.basename(img_path)
        
        # Load and preprocess using rasterio wrapper
        img_np, raster_transform, src_crs_epsg = load_and_preprocess_raster(
            img_path, target_crs_epsg=32643, target_resolution=1.0, tile_size=512
        )
        
        # Run inference
        road_mask = predict_mask(model, img_np, device=device, threshold=args.threshold)
        road_pixels = int(road_mask.sum())
        
        # Skeletonize to graph
        tile_graph = mask_to_graph(
            binary_mask=road_mask,
            raster_transform=raster_transform,
            source_crs_epsg=32643 if src_crs_epsg == -1 else src_crs_epsg,
            target_utm_epsg=32643
        )
        
        # Rename nodes to be unique across tiles
        mapping = {node: f"tile_{idx}_node_{node}" for node in tile_graph.nodes()}
        tile_graph = nx.relabel_nodes(tile_graph, mapping)
        
        # Merge to global unhealed graph
        combined_unhealed_graph.add_nodes_from(tile_graph.nodes(data=True))
        combined_unhealed_graph.add_edges_from(tile_graph.edges(data=True))
        
        tile_summaries.append({
            "name": img_name,
            "road_pixels": road_pixels,
            "nodes": tile_graph.number_of_nodes(),
            "edges": tile_graph.number_of_edges(),
            "status": "Processed"
        })
        print(f"  [OK] {img_name}: Extracted {tile_graph.number_of_nodes()} nodes, {tile_graph.number_of_edges()} edges.")

    # 5. Spatial Stitching and Graph Healing
    print("\n3. Stitching boundaries and healing topology...")
    stitched_graph = spatial_stitch_nodes(combined_unhealed_graph, tolerance_m=2.0)
    healed_graph = heal_graph_connectivity(stitched_graph)
    
    print("\n4. Running centrality analysis...")
    centrality = compute_betweenness_centrality(healed_graph)
    
    # 6. Export outputs
    print("\n5. Exporting outputs to outputs/local_validation/ ...")
    
    # Save Pickles
    unhealed_pkl_path = os.path.join(outputs_dir, "unhealed_graph.pkl")
    healed_pkl_path = os.path.join(outputs_dir, "healed_graph.pkl")
    with open(unhealed_pkl_path, "wb") as f:
        pickle.dump(combined_unhealed_graph, f)
    with open(healed_pkl_path, "wb") as f:
        pickle.dump(healed_graph, f)
        
    # Save centrality JSON
    centrality_json_path = os.path.join(outputs_dir, "centrality.json")
    centrality_str_keys = {str(k): float(v) for k, v in centrality.items()}
    with open(centrality_json_path, "w") as f:
        json.dump(centrality_str_keys, f, indent=2)
        
    # Export criticality geojson
    geojson_path = os.path.join(outputs_dir, "criticality.geojson")
    export_criticality_geojson(healed_graph, centrality, geojson_path)
    
    # Save metric summaries
    metrics_summary_path = os.path.join(outputs_dir, "metrics.json")
    iou_scores = []
    dice_scores = []
    
    for img_path, mask_path in valid_images:
        if mask_path:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                gt_mask = (gt_mask > 127).astype(np.uint8)
                
                # Preprocess image
                img_np, _, _ = load_and_preprocess_raster(img_path, tile_size=512)
                pred_mask = predict_mask(model, img_np, device=device, threshold=args.threshold)
                
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = (intersection + 1e-8) / (union + 1e-8)
                dice = (2.0 * intersection + 1e-8) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
                
                iou_scores.append(float(iou))
                dice_scores.append(float(dice))
                
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    metrics_summary = {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "nodes_before_healing": combined_unhealed_graph.number_of_nodes(),
        "edges_before_healing": combined_unhealed_graph.number_of_edges(),
        "nodes_after_healing": healed_graph.number_of_nodes(),
        "edges_after_healing": healed_graph.number_of_edges()
    }
    with open(metrics_summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 7. Generate comparison plots using first valid image
    print("\n6. Generating diagnostic comparison plots...")
    test_img_path, test_mask_path = valid_images[0]
    test_img_np, test_transform, _ = load_and_preprocess_raster(test_img_path, tile_size=512)
    test_pred_mask = predict_mask(model, test_img_np, device=device, threshold=args.threshold)
    
    # Save mask_check_5.png
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img_np)
    axes[0].set_title("Original Satellite Image")
    axes[0].axis("off")
    
    axes[1].imshow(test_pred_mask, cmap="gray")
    axes[1].set_title("Predicted Binary Mask")
    axes[1].axis("off")
    
    # Overlay predicted roads in red
    overlay = test_img_np.copy()
    overlay[test_pred_mask == 1] = [255, 50, 50]  # Red roads
    axes[2].imshow(overlay)
    axes[2].set_title("Road Mask Overlay (Red)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt_path_mask = os.path.join(outputs_dir, "mask_check_5.png")
    plt.savefig(plt_path_mask, dpi=150)
    plt.close()
    
    # Save diagnostic_occlusion_mask_check.png with synthetic occlusion added
    from data_pipeline import add_synthetic_occlusion
    occ_img, occ_mask = add_synthetic_occlusion(test_img_np, test_pred_mask, occlusion_type="shadow", return_mask=True)
    
    # Re-normalize for model prediction
    val_aug = get_val_augmentations()
    aug_res = val_aug(image=occ_img)
    tensor_img = aug_res["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        occ_pred_mask = (probs > args.threshold).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(occ_img)
    axes[0, 0].set_title("(a) Input Image with Occlusion")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(occ_mask > 0.1, cmap="gray")
    axes[0, 1].set_title("(b) Occlusion Mask (Threshold > 0.1)")
    axes[0, 1].axis("off")
    
    if test_mask_path:
        gt_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        axes[1, 0].imshow(gt_mask, cmap="gray")
    else:
        axes[1, 0].imshow(test_pred_mask, cmap="gray")
    axes[1, 0].set_title("(c) Ground Truth Mask")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(occ_pred_mask, cmap="gray")
    axes[1, 1].set_title("(d) Predicted Mask under Occlusion")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt_path_diag = os.path.join(outputs_dir, "diagnostic_occlusion_mask_check.png")
    plt.savefig(plt_path_diag, dpi=150)
    plt.close()
    
    print(f"  [OK] Saved outputs/local_validation/mask_check_5.png")
    print(f"  [OK] Saved outputs/local_validation/diagnostic_occlusion_mask_check.png")

    print("\n" + "=" * 60)
    print("LOCAL VALIDATION COMPLETE!")
    print("=" * 60)
    print(f"Mean IoU: {mean_iou:.4f} | Mean Dice: {mean_dice:.4f}")
    print(f"Stitched Graph: {healed_graph.number_of_nodes()} nodes, {healed_graph.number_of_edges()} edges (healed).")
    print("=" * 60)

import pickle
if __name__ == "__main__":
    main()
