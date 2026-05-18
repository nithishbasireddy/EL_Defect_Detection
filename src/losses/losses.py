import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Class weights (VERY IMPORTANT)
# Order: background, dark, cross, crack, busbar
class_weights = torch.tensor([0.05, 1.5, 2.0, 5.0, 1.0])

# Dice Loss
dice_loss = smp.losses.DiceLoss(mode='multiclass')

# Final loss
def loss_fn(pred, target):
    device = pred.device
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)