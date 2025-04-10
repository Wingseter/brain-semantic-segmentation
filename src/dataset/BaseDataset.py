from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, data_path: str, train_transform, val_transform):
        super().__init__()
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform

    @abstractmethod
    def train_dataset(self):
        """
        This method should be implemented in the subclass to return the training dataset.
        """
        pass

    @abstractmethod
    def val_dataset(self):
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
