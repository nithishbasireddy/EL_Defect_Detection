import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():

    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.RandomBrightnessContrast(p=0.3),

        A.Normalize(),
        ToTensorV2()
    ])


def get_val_transforms():

    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])