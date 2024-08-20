from torch.utils.data import DataLoader
import lightning as L
from importlib import import_module

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        num_workers: int,
        transform_name: str, 
        dataset_name: str, 
        dataset_params
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers,
        self.transform_name = transform_name,
        self.dataset_name = dataset_name,
        self.dataset_params = dataset_params

        # I Dont know why but it changes to tuple maybe its a bug if hydra?
        if type(self.num_workers) == tuple:
            self.num_workers = self.num_workers[0]
        
        augment_module = getattr(import_module("src.augmentation"), transform_name)
        
        # setup transform
        self.train_transform = augment_module()
        self.val_transform = augment_module()

        # Prepare the dataset initialization parameters
        dataset_params = dict(dataset_params)  # Convert to a plain dictionary if needed

        # Add transforms to dataset_params
        dataset_params['train_transform'] = self.train_transform
        dataset_params['val_transform'] = self.val_transform

        # Dynamically import the dataset module and get the dataset class
        dataset_class = getattr(import_module("src.dataset"), dataset_name)
        # Initialize the dataset class with the parameters using **kwargs
        self.brain_dataset = dataset_class(**dataset_params)


    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = self.brain_dataset.train_dataset()
            self.val_data = self.brain_dataset.val_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = self.brain_dataset.test_dataset()

    def train_dataloader(self):
        print("num Workers train_dataloader" ,self.num_workers)
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    