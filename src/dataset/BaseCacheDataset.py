
from typing import Any
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset

class BaseDataset(CacheDataset):
    def __init__(self):
        super.__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)