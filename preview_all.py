import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("="*60)
    print("HACKATHON IMAGES PREVIEW GENERATOR")
    print("="*60)
    
    # 1. Paths
    image_pattern = os.path.join(PROJECT_ROOT, "hacthon_req", "*.jpeg")
    output_path = os.path.join(PROJECT_ROOT, "outputs", "all_images_preview.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Find and sort all images (1.jpeg to 7.jpeg)
    image_files = sorted(glob.glob(image_pattern))
    if not image_files:
        print(f"[ERROR] No images found matching: {image_pattern}")
        return
        
    print(f"Found {len(image_files)} images to preview.")
    
    # 2. Setup grid dimensions (2 rows, 4 columns for 7 images)
    rows = 2
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    
    # Flatten axes array for easy indexing
    axes_flat = axes.flatten()
    
    # 3. Load, resize and display each image
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"   Processing {img_name}...")
        
        # Load and resize to 300x300
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Display on the current subplot
        ax = axes_flat[idx]
        ax.imshow(img_resized)
        ax.axis("off")
        
        # Label the filename below the thumbnail
        ax.text(0.5, -0.1, img_name, ha="center", va="top", 
                transform=ax.transAxes, fontsize=12, fontweight="bold")
                
    # 4. Hide unused subplots (the 8th cell is empty)
    for idx in range(len(image_files), len(axes_flat)):
        axes_flat[idx].axis("off")
        
    # 5. Save the preview
    print("\nSaving preview composite image...")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*60)
    print("OK: PREVIEW COMPOSITE IMAGE SAVED TO:")
    print(f"    {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
