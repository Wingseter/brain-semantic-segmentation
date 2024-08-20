import lightning as L
import torch
from monai.losses import DiceLoss
from monai.networks.nets import *
from monai.networks.layers import *
from monai.metrics import compute_generalized_dice
from monai.inferers import SimpleInferer, SlidingWindowInferer

from importlib import import_module

class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "BaseModel",
        learning_rate: float = 1e-2,
        use_scheduler: bool = True,    
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        select_model = getattr(import_module("src.models"), model_name)
        self._model = select_model()

        self.criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    def forward(self, x):
        return self._model(x)
    
    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step =True, on_epoch=True, prog_bar=True)
        return loss
    
    def _inference(self, input):
        def _compute(input):
            inferer =  SimpleInferer()
            return inferer(
                inputs = input,     
                network = self._model,
            )
        
        VAL_AMP = True
        if VAL_AMP:
            with torch.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
        
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        print(y.shape)
        print(logits.shape)
        loss = self.criterion(logits, y)
        y_pred = self._inference(x)
        dice = compute_generalized_dice(y_pred, y)
        dice = dice.mean() if len(dice) > 0 else dice
        self.log("val_loss", loss, on_step = True, on_epoch = True, prob_bar = True)
        self.log("val_dice", dice, on_step = True, on_epoch = True, prob_bar = True)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=self.learning_rate, weight_decay=0.05
        # )
        optimizer = torch.optim.Adam(self.parameters(), 1e-4, weight_decay=1e-5)

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        if self.use_scheduler:
            # Add lr scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)
            configuration["lr_scheduler"] = scheduler

        return configuration
    