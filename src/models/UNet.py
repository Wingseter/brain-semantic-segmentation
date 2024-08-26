from .BaseModel import BaseModel
from monai.networks.nets import unet
from monai.networks.layers import *

class UNet(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        self.segResnet = unet(
            **model_params
        )

    def forward(self, x):
        y = self.segResnet(x)
        return y