import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Allow importing src/ modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from segmentation_train import build_model, predict_mask

def main():
    print("="*60)
    print("ROAD SEGMENTATION MASK CHECKER")
    print("="*60)
    
    # 1. Paths
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    image_path = os.path.join(PROJECT_ROOT, "hacthon_req", "5.jpeg")
    output_path = os.path.join(PROJECT_ROOT, "outputs", "mask_check_5.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Model weights not found at: {checkpoint_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found at: {image_path}")
        return
        
    # 2. Load Model
    print("1. Loading trained segmentation model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = build_model(
        architecture="Unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
        
    # 3. Load Image
    print("2. Loading and preparing image hacthon_req/5.jpeg...")
    img_pil = Image.open(image_path).convert("RGB")
    # Resize to match model training resolution (512x512)
    img_pil = img_pil.resize((512, 512), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil)
    
    # 4. Inference
    print("3. Running model inference...")
    road_mask = predict_mask(model, img_np, device=device, threshold=0.5)
    print(f"   Inference complete. Detected {int(road_mask.sum())} road pixels.")
    
    # 5. Create Overlaid Image
    print("4. Creating blended red overlay...")
    overlaid = img_np.copy()
    red_mask = np.zeros_like(img_np)
    red_mask[road_mask == 1] = [255, 0, 0]
    # 60% original image + 40% red overlay for blended look
    overlaid = np.where(road_mask[:, :, np.newaxis] == 1, 
                        (img_np * 0.6 + red_mask * 0.4).astype(np.uint8), 
                        img_np)
                        
    # 6. Save side-by-side plot using Matplotlib
    print("5. Generating side-by-side visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image (5.jpeg)", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Predicted Binary Mask
    axes[1].imshow(road_mask, cmap="gray")
    axes[1].set_title("Predicted Binary Mask", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    # Mask Overlaid on Original (Red)
    axes[2].imshow(overlaid)
    axes[2].set_title("Road Mask Overlay (Red)", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*60)
    print(f"OK: VISUALIZATION SAVED SUCCESSFULLY TO:")
    print(f"    {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
