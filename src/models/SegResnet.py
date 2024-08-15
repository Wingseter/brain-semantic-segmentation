from BaseModel import BaseModel
from monai.networks.nets import SegResNet
from monai.networks.layers import *
from monai.networks.layers import Norm

class SegResnet_option1(BaseModel):
    def __init__(self):
        super().__init__()
        self.segResnet = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )

    def forward(self, x):
        y = self.segResnet(x)
        return 