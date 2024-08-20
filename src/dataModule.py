from torch.utils.data import DataLoader

from importlib import import_module

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, data_path: str, augmentation: str, dataset_name: str, stage=None) -> None: 
        augment_module = getattr(import_module("augmentation"), augmentation)
        # Training transform
        train_transform = augment_module()
        # Validation transform
        val_transform = augment_module()

        DatasetClass = getattr(import_module("dataset"), dataset_name)
        # Initialize the class with the parameters
        brain_dataset = DatasetClass()


        self.train_data = brain_dataset.train_dataset()
        self.val_data = brain_dataset.valid_dataset()
        self.test_data = brain_dataset.test_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    