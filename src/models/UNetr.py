from .BaseModel import BaseModel
from monai.networks.nets import unetr
from monai.networks.layers import *

class UNetr(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        self.unetr = UNetr(
            **model_params
        )

    def forward(self, x):
        y = self.unetr(x)
        return y