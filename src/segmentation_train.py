"""
Phase 2: Segmentation Model Training & Evaluation
Maps to:
  - 4 Core Objectives:
    * Occlusion-Aware Extraction (U-Net / DeepLabV3+ with ResNet backbone)
  - Evaluation Parameters:
    * IoU & Dice Score
    * Occlusion-Recall (IoU computed only on synthetically occluded regions)
    * Generalisation (cross-terrain evaluation)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RoadDataset(Dataset):
    """
    PyTorch Dataset for road segmentation tiles.

    Expects pre-tiled numpy arrays (HxWxC images, HxW masks).
    Supports optional Albumentations augmentation pipeline.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        augmentations: Optional[A.Compose] = None,
        apply_occlusion: bool = False,
    ):
        """
        Args:
            images: List of HxWxC uint8 image tiles.
            masks: List of HxW uint8 binary mask tiles (1=road, 0=bg).
            augmentations: Albumentations Compose pipeline (applied to both image & mask).
            apply_occlusion: If True, randomly applies synthetic occlusion from data_pipeline.
        """
        assert len(images) == len(masks), "images and masks must be the same length"
        self.images = images
        self.masks = masks
        self.augmentations = augmentations
        self.apply_occlusion = apply_occlusion

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Optionally apply synthetic occlusion (shadow/canopy/cloud) for robustness
        if self.apply_occlusion and np.random.random() < 0.5:
            from data_pipeline import add_synthetic_occlusion
            occ_type = np.random.choice(["shadow", "canopy", "cloud"])
            image = add_synthetic_occlusion(image, mask, occlusion_type=occ_type)

        if self.augmentations:
            result = self.augmentations(image=image, mask=mask)
            image = result['image']       # already a tensor from ToTensorV2
            mask = result['mask']
        else:
            # Manual conversion: HWC uint8 -> CHW float32
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Ensure mask has a channel dimension (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask}


# ---------------------------------------------------------------------------
# Default augmentations
# ---------------------------------------------------------------------------

def get_train_augmentations(tile_size: int = 512) -> A.Compose:
    """Standard training augmentations for satellite road imagery."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_augmentations() -> A.Compose:
    """Validation augmentations (normalize only, no geometric transforms)."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    architecture: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Builds a segmentation model using segmentation_models_pytorch.

    Args:
        architecture: One of 'Unet', 'UnetPlusPlus', 'DeepLabV3Plus'.
        encoder_name: Backbone encoder (e.g. 'resnet34', 'resnet50', 'efficientnet-b3').
        encoder_weights: Pretrained weights ('imagenet' or None).
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output classes (1 for binary road/background).
    """
    arch_map = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
    }
    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {list(arch_map.keys())}")

    model = arch_map[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # raw logits; we apply sigmoid in the loss / inference
    )
    return model


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes IoU, Dice, Precision, Recall for binary segmentation.

    Args:
        preds: Raw logits (B, 1, H, W).
        targets: Binary masks (B, 1, H, W).
        threshold: Binarization threshold applied after sigmoid.

    Returns:
        Dict with 'iou', 'dice', 'precision', 'recall'.
    """
    probs = torch.sigmoid(preds)
    binary = (probs > threshold).float()

    tp = (binary * targets).sum().item()
    fp = (binary * (1 - targets)).sum().item()
    fn = ((1 - binary) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Trains the segmentation model and saves the best checkpoint by validation IoU.

    Args:
        model: The segmentation model (from build_model).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate for AdamW.
        checkpoint_dir: Directory to save model checkpoints.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Dict of training history: {'train_loss', 'val_loss', 'val_iou', 'val_dice'}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [],
    }
    best_iou = 0.0

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        agg_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                metrics = compute_metrics(logits, masks)
                for k in agg_metrics:
                    agg_metrics[k] += metrics[k] * images.size(0)
                n_val += images.size(0)

        val_loss /= n_val
        for k in agg_metrics:
            agg_metrics[k] /= n_val

        history["val_loss"].append(val_loss)
        history["val_iou"].append(agg_metrics["iou"])
        history["val_dice"].append(agg_metrics["dice"])

        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"IoU: {agg_metrics['iou']:.4f} | Dice: {agg_metrics['dice']:.4f}"
        )

        # Save best model
        if agg_metrics["iou"] > best_iou:
            best_iou = agg_metrics["iou"]
            path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            logger.info(f"  ↳ New best IoU={best_iou:.4f}, saved to {path}")

    # Save final model regardless
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    return history


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Runs inference on a single HxWxC uint8 image tile and returns a binary mask.

    Args:
        model: Trained segmentation model.
        image: HxWxC uint8 image.
        device: 'cuda' or 'cpu' (auto-detect if None).
        threshold: Probability threshold for binarization.

    Returns:
        HxW uint8 binary mask (0 or 1).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    aug = get_val_augmentations()
    result = aug(image=image)
    tensor = result["image"].unsqueeze(0).to(device)  # (1, C, H, W)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    return mask
