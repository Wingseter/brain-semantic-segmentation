from monai.transforms import (
    Activations,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    AsDiscrete,
    Resized,

)

class BaseAugmentation:
    def __init__(self, resize, mean, std, **kwargs):
        self.transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)