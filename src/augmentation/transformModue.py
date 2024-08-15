import torch
from monai.transforms import (
    MapTransform,
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert input channels to specific output channels using PyTorch tensors.

    Input Dimension: (W, H, D, C(Channel)) 
    Input Channel Description:
        0: 'Necrotic (NEC)' unique (0, 1)
        1: 'Edema (ED)' unique (0, 1)
        2: 'Tumour (ET)' unique (0, 1)

    Output Dimension: (C(Channel), W, H, D)
    Output Channel Description:
        0: TC (Tumor core)
        1: WT (Whole tumor)
        2: ET (Enhancing tumor)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Convert data to PyTorch tensor if not already one
            data_tensor = data[key]

            necrotic = data_tensor[..., 0]
            edema = data_tensor[..., 1]
            enhancing = data_tensor[..., 2]

            # Compute TC: necrotic and enhancing tumor areas (logical OR)
            tc = torch.logical_or(necrotic, enhancing)

            # Compute WT: all tumor areas (logical OR among all channels)
            wt = torch.logical_or(torch.logical_or(necrotic, edema), enhancing)

            # Combine channels into a new tensor with dimensions (C, W, H, D)
            d[key] = torch.stack([tc, wt, enhancing], dim=0)

        return d
