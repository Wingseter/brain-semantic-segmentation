from torch.utils.data import DataLoader

from importlib import import_module

class DataModule(L.LightningDataModule):
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
        augment_module = getattr(import_module("augmentation"),'BratsAugmentation')
        # Training transform
        train_transform = augment_module()
        # Validation transform
        val_transform = augment_module()

        dataset_module = getattr(import_module("dataset"), "")

        self.train_data = CacheDataset(data=train_dict, transform=train_transform, cache_rate=0.1, num_workers=10)
        self.val_data = CacheDataset(data=valid_dict, transform=val_transform, cache_rate=0.1, num_workers=10)

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