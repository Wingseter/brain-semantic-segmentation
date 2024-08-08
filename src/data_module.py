import lightning as L
from monai.transforms import (
    Compose,
    CropForegraound,

)
from torch.utils.data import DataLoader

class BrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        task: str,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed


    def setup(self, stage=None):
        # Training transform
        train_transform = Compose(
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

        # Validation transform
        val_transform = Compose(
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

        self.train_data = CacheDataset(data=train_dict, transform=train_transform, cache_rate=0.1, num_workers=3)
        self.val_data = CacheDataset(data=valid_dict, transform=val_transform, cache_rate=0.1, num_workers=3)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )