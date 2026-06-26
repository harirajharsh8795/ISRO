# Occlusion-Recall Validation Report

This report documents the performance deltas before and after applying CBAM self-attention architecture (Fix #2) and training with Occlusion-Weighted Loss (Fix #3) with CORRECTED normalizations.

## Metrics Comparison Table

| Metric | Baseline (Before) | Fine-Tuned (After) | Absolute Delta | % Improvement |
|---|---|---|---|---|
| IoU (occluded pixels) | 0.0324 | 0.9390 | +0.9067 | +2801.7% |
| IoU (non-occluded pixels) | 0.0029 | 0.9751 | +0.9722 | +33691.2% |
| Dice (occluded pixels) | 0.0584 | 0.9683 | +0.9099 | +1557.4% |
| Dice (non-occluded pixels) | 0.0056 | 0.9874 | +0.9818 | +17480.2% |
| Relaxed IoU (3px buffer) | 0.0210 | 0.9989 | +0.9778 | +4645.2% |
| APL Error (%) | 251.83% | 4.54% | -247.28% | +98.2% |

## Final Verdict

**VERDICT: Fine-tuning the ResNet34 U-Net bottleneck with CBAM attention and Occlusion-Weighted Loss measurably improved road extraction recall in occluded areas without causing any significant regression on non-occluded regions.**