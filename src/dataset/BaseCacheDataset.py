
from typing import Any
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset

class BaseDataset(CacheDataset):
    def __init__(self, img_path, transform):
        super().__init__()
        self.img_paths = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)