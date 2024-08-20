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
        self.train_batch_size = batch_size
        self.valid_batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, augmentation_name: str, dataset_name: str, dataset_params, stage=None) -> None: 
        augment_module = getattr(import_module("augmentation"), augmentation_name)
        # setup transform
        self.train_transform = augment_module()
        self.val_transform = augment_module()

        DatasetClass = getattr(import_module("dataset"), dataset_name)

                # Prepare the dataset initialization parameters
        dataset_params = dict(dataset_params)  # Convert to a plain dictionary if needed

        # Add transforms to dataset_params
        dataset_params['train_transform'] = self.train_transform
        dataset_params['valid_transform'] = self.val_transform

        # Dynamically import the dataset module and get the dataset class
        DatasetClass = getattr(import_module("dataset"), dataset_name)

        # Initialize the dataset class with the parameters using **kwargs
        brain_dataset = DatasetClass(**dataset_params)

        self.train_data = brain_dataset.train_dataset()
        self.val_data = brain_dataset.valid_dataset()
        self.test_data = brain_dataset.test_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    