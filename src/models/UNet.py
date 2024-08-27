from .BaseModel import BaseModel
from monai.networks.nets import unet
from monai.networks.layers import *

class UNet(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        self.unet = UNet(
            **model_params
        )

    def forward(self, x):
        y = self.unet(x)
        return y