import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from .BaseCacheDataset import BaseCacheDataset
from monai.data.dataset import CacheDataset

class BrainDataset(BaseCacheDataset):
    def __init__(self, 
                 data_path,
                 cache_rate, 
                 num_workers, 
                 train_valid_test_split_rate,
                 train_transform, 
                 val_transform):
        super().__init__(\
            data_path,
            cache_rate, 
            num_workers,
            train_transform, 
            val_transform)
        
        self.train_valid_test_split_rate = train_valid_test_split_rate
        self.num_workers = num_workers

        self.image_path = Path(self.data_path) / "image"
        self.label_path = Path(self.data_path) / "label"
        self.train_data_dicts = []
        self.valid_data_dicts = []
        self.test_data_dicts = []

        image_files = sorted([self.image_path / file_path for file_path in os.listdir(self.image_path)])

        for image_file in image_files:
            match_number = image_file.name.split('_')[-2]
            label_file = self.label_path / f"volume_{match_number}_mask.nii"

            if os.path.exists(label_file):
                self.train_data_dicts.append({"image": str(image_file), "label": str(label_file)})

        # Split the data
        self._split_data(self.train_valid_test_split_rate)


    def _split_data(self, split_rate):
        """
        Split the dataset into training, validation, and testing sets based on the provided split_rate.

        Parameters:
        - split_rate: Tuple of (train_rate, valid_rate, test_rate) indicating the split proportions.
        """
        train_rate, valid_rate, test_rate = split_rate

        train_data, temp_data = train_test_split(self.train_data_dicts, test_size=(1 - train_rate))
        valid_data, test_data = train_test_split(temp_data, test_size=(test_rate / (valid_rate + test_rate)))

        self.train_data_dicts = train_data
        self.valid_data_dicts = valid_data
        self.test_data_dicts = test_data

    def train_dataset(self):
        """
        Returns a CacheDataset for the training data.
        """
        return CacheDataset(
            data=self.train_data_dicts, 
            transform=self.train_transform, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
        )

    def val_dataset(self):
        """
        Returns a CacheDataset for the validation data.
        """
        return CacheDataset(
            data=self.valid_data_dicts, 
            transform=self.val_transform, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
            )

    def test_dataset(self):
        """
        Returns a CacheDataset for the test data.
        """
        return CacheDataset(data=self.test_data_dicts, transform=self.valid_transform, cache_rate=self.cache_rate, num_workers=self.num_workers)
