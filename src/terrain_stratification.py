import numpy as np
import cv2
from typing import Dict, List, Any
import networkx as nx

def classify_terrain_tile(image: np.ndarray) -> str:
    """
    Classifies an RGB tile into 'urban', 'forested', or 'rural' using color
    and texture heuristics.
    """
    # Convert to float for calculation
    img = image.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    
    # NDVI-like proxy: Green-Red Ratio (GRR)
    grr = (g - r) / (g + r + 1e-8)
    mean_grr = np.mean(grr)
    
    # Texture proxy: variance of Laplacian gradients (measures edge density)
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
    unhealed_graphs: List[nx.Graph],
    healed_graphs: List[nx.Graph],
) -> Dict[str, Dict[str, float]]:
    """
    Groups evaluation metrics by terrain category and returns a comparison dictionary.
    """
    from segmentation_train import compute_metrics
    from centrality import compute_connectivity_ratio
    
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
