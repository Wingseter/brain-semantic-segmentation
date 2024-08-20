from typing import Any
from abc import ABC, abstractmethod
from dataset.BaseDataset import BaseDataset

class BaseCacheDataset(BaseDataset, ABC):
    def __init__(self, data_path: str, train_transform, valid_transform, cache_rate: float, num_workers: int):
        super().__init__(data_path, train_transform=train_transform, valid_transform=valid_transform)
        self.cache_rate = cache_rate
        self.num_workers = num_workers

    @abstractmethod
    def train_dataset(self):
        """
        This method should be implemented in the subclass to return the training dataset.
        """
        pass

    @abstractmethod
    def valid_dataset(self):
        """
        This method should be implemented in the subclass to return the validation dataset.
        """
        pass
    
    @abstractmethod
    def test_dataset(self):
        """
        This method should be implemented in the subclass to return the test dataset.
        """
        pass
