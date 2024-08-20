from .BaseModel import BaseModel
from monai.networks.nets import SegResNet
from monai.networks.layers import *
from monai.networks.layers import Norm

class SegResnet(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        self.segResnet = SegResNet(
            **model_params
        )

    def forward(self, x):
        y = self.segResnet(x)
        return y