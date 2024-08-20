from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
)

from src.augmentation.BaseAugmentation import BaseAugmentation
from src.augmentation.transformModue import ConvertToMultiChannelBasedOnBratsClassesd

class BratsAugmentation_base(BaseAugmentation):
    def __init__(self, **kwargs):
        self.transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Resized(keys=["image"], spatial_size=[128, 128, 80], mode="bilinear"),
                Resized(keys=["label"], spatial_size=[128, 128, 80], mode="nearest"),
            ]
        )

    def __call__(self):
        return self.transform