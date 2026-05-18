import segmentation_models_pytorch as smp


def get_loss():

    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    focal_loss = smp.losses.FocalLoss(
        mode="multiclass",
        gamma=2.0
    )

    def loss_fn(pred, target):
        return dice_loss(pred, target) + focal_loss(pred, target)

    return loss_fn