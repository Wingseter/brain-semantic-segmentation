
from BaseDataset import BaseDataset

class BrainDataset(BaseDataset):
    def __init__(self, img_path, transform):
        super().__init__(img_path, transform)
    



    def __getitem__(self, index):
        image = self.img_paths[index]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)