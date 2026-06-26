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
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# CBAM Attention Modules & Wrapper
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class AttentionUnetWrapper(nn.Module):
    """
    Wraps an SMP model (like Unet or UnetPlusPlus) to run a CBAM self-attention
    module at the bottleneck (the bridge between the encoder and decoder).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        # Dynamically determine out channels of the encoder bottleneck
        bottleneck_channels = base_model.encoder.out_channels[-1]
        self.cbam = CBAM(bottleneck_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encoder(x)
        # Apply self-attention at the bottleneck feature map (last encoder output)
        features[-1] = self.cbam(features[-1])
        decoder_output = self.base_model.decoder(features)
        masks = self.base_model.segmentation_head(decoder_output)
        return masks

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        """Maps keys dynamically to self.base_model if the checkpoint has no base_model prefix."""
        new_state_dict = {}
        has_prefix = any(k.startswith("base_model.") for k in state_dict.keys())
        if not has_prefix:
            # Add prefix for wrapped parameters
            for k, v in state_dict.items():
                new_state_dict[f"base_model.{k}"] = v
            # Retain any new attention weights already present in state_dict
            for k, v in state_dict.items():
                if k.startswith("cbam."):
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict
        return super().load_state_dict(new_state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RoadDataset(Dataset):
    """
    PyTorch Dataset for road segmentation tiles.
    Expects pre-tiled numpy arrays (HxWxC images, HxW masks).
    Supports optional Albumentations augmentation pipeline and synthetic occlusion returning.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        augmentations: Optional[A.Compose] = None,
        apply_occlusion: bool = False,
    ):
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
        h, w = image.shape[:2]
        
        # Initialize zero occlusion mask
        occlusion_mask = np.zeros((h, w), dtype=np.float32)

        # Optionally apply synthetic occlusion (shadow/canopy/cloud) for robustness
        if self.apply_occlusion and np.random.random() < 0.5:
            from data_pipeline import add_synthetic_occlusion
            occ_type = np.random.choice(["shadow", "canopy", "cloud"])
            image, occ_m = add_synthetic_occlusion(image, mask, occlusion_type=occ_type, return_mask=True)
            occlusion_mask = occ_m.astype(np.float32)

        if self.augmentations:
            result = self.augmentations(image=image, mask=mask)
            image = result['image']       # already a tensor from ToTensorV2
            mask = result['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Convert occlusion mask to tensor
        occ_tensor = torch.from_numpy(occlusion_mask).unsqueeze(0).float()

        return {"image": image, "mask": mask, "occlusion_mask": occ_tensor}


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
    attention: bool = False,
) -> nn.Module:
    """
    Builds a segmentation model using segmentation_models_pytorch, with optional CBAM bottleneck.
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
        activation=None,  # raw logits
    )
    
    # Wrap model with CBAM bottleneck attention if requested
    if attention:
        if architecture in ["Unet", "UnetPlusPlus"]:
            model = AttentionUnetWrapper(model)
            logger.info("Wrapped model with CBAM bottleneck self-attention layer.")
        else:
            logger.warning(f"Bottleneck CBAM wrapper not supported directly for {architecture}. Returning base model.")

    return model


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, *args) -> torch.Tensor:
        targets = targets.float()
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


class OcclusionWeightedDiceBCELoss(nn.Module):
    """
    BCE + Dice loss that applies higher pixel weights to regions covered by the
    synthetic occlusion mask, forcing focus on reconstructing occluded road segments.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0, multiplier: float = 3.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.multiplier = multiplier

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, occlusion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets = targets.float()
        
        # Build spatial weights mapping
        if occlusion_mask is not None:
            occlusion_mask = occlusion_mask.float()
            weight_map = 1.0 + (self.multiplier - 1.0) * occlusion_mask
        else:
            weight_map = torch.ones_like(targets)

        # Weighted Binary Cross-Entropy
        import torch.nn.functional as F
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = (bce_loss * weight_map).mean()

        # Weighted Dice
        probs = torch.sigmoid(logits)
        weighted_intersection = (probs * targets * weight_map).sum()
        weighted_union = (probs * weight_map).sum() + (targets * weight_map).sum()
        
        dice_loss = 1.0 - (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)

        return self.bce_weight * weighted_bce + (1.0 - self.bce_weight) * dice_loss


# ---------------------------------------------------------------------------
# Metrics (Relaxed IoU and Occlusion-Recall)
# ---------------------------------------------------------------------------

def relaxed_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, buffer_px: int = 3) -> float:
    """
    Computes relaxed IoU where predicted pixels within buffer_px distance of
    ground-truth count as True Positives.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # Dilate ground truth to capture relaxed TP
    gt_dilated = binary_dilation(gt, iterations=buffer_px)
    tp_relaxed = pred & gt_dilated
    fp_relaxed = pred & (~gt_dilated)

    # Dilate prediction to capture relaxed FN
    pred_dilated = binary_dilation(pred, iterations=buffer_px)
    fn_relaxed = gt & (~pred_dilated)

    tp = np.sum(tp_relaxed)
    fp = np.sum(fp_relaxed)
    fn = np.sum(fn_relaxed)

    return float(tp / (tp + fp + fn + 1e-8))


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard IoU, Dice, Precision, Recall, and Relaxed IoU.
    """
    targets_f = targets.float()
    probs = torch.sigmoid(preds)
    binary = (probs > threshold).float()

    tp = (binary * targets_f).sum().item()
    fp = (binary * (1 - targets_f)).sum().item()
    fn = ((1 - binary) * targets_f).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # Compute relaxed IoU over the batch using numpy
    batch_relaxed_iou = 0.0
    pred_np = binary.squeeze(1).cpu().numpy().astype(np.uint8)
    gt_np = targets.squeeze(1).cpu().numpy().astype(np.uint8)
    
    if pred_np.ndim == 2:  # Single item batch
        batch_relaxed_iou = relaxed_iou(pred_np, gt_np, buffer_px=3)
    else:
        for i in range(pred_np.shape[0]):
            batch_relaxed_iou += relaxed_iou(pred_np[i], gt_np[i], buffer_px=3)
        batch_relaxed_iou /= pred_np.shape[0]

    return {
        "iou": iou, 
        "dice": dice, 
        "precision": precision, 
        "recall": recall,
        "relaxed_iou": batch_relaxed_iou
    }


def compute_occlusion_recall_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    occlusion_mask: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Calculates standard segmentation metrics isolated separately on 
    occluded regions and non-occluded regions.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    occ = occlusion_mask.astype(bool)

    # Subset pixels
    pred_occ = pred & occ
    gt_occ = gt & occ

    pred_non_occ = pred & (~occ)
    gt_non_occ = gt & (~occ)

    def _get_stats(p, g):
        tp = np.sum(p & g)
        fp = np.sum(p & (~g))
        fn = np.sum((~p) & g)
        
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        
        return {"iou": float(iou), "dice": float(dice), "precision": float(prec), "recall": float(rec)}

    return {
        "occluded": _get_stats(pred_occ, gt_occ),
        "non_occluded": _get_stats(pred_non_occ, gt_non_occ)
    }


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
    use_occlusion_weighted_loss: bool = False,
) -> Dict[str, List[float]]:
    """
    Trains the segmentation model and saves the best checkpoint by validation IoU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    
    if use_occlusion_weighted_loss:
        criterion = OcclusionWeightedDiceBCELoss(multiplier=3.0)
        logger.info("Using OcclusionWeightedDiceBCELoss (3.0x weight on occluded pixels)")
    else:
        criterion = DiceBCELoss()
        logger.info("Using standard DiceBCELoss")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_relaxed_iou": []
    }
    best_iou = 0.0

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            if use_occlusion_weighted_loss:
                occ_masks = batch["occlusion_mask"].to(device)
                logits = model(images)
                loss = criterion(logits, masks, occ_masks)
            else:
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
        agg_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "relaxed_iou": 0.0}
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                if use_occlusion_weighted_loss:
                    occ_masks = batch["occlusion_mask"].to(device)
                    loss = criterion(logits, masks, occ_masks)
                else:
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
        history["val_relaxed_iou"].append(agg_metrics["relaxed_iou"])

        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"IoU: {agg_metrics['iou']:.4f} | Relaxed IoU: {agg_metrics['relaxed_iou']:.4f} | Dice: {agg_metrics['dice']:.4f}"
        )

        # Save best model
        if agg_metrics["iou"] > best_iou:
            best_iou = agg_metrics["iou"]
            path = os.path.join(checkpoint_dir, "best_model.pth")
            # Unbox model from wrapper if saving state dict directly (for compatibility)
            state = model.state_dict()
            torch.save(state, path)
            logger.info(f"  ↳ New best IoU={best_iou:.4f}, saved to {path}")

    # Save final model
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
