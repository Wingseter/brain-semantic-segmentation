import lightning as L
import torch
from monai.losses import DiceLoss
from monai.networks.nets import *
from monai.networks.layers import *
from monai.metrics import compute_generalized_dice
from monai.inferers import SimpleInferer, SlidingWindowInferer
import torch.nn.functional as F

from importlib import import_module

class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "BaseModel",
        learning_rate: float = 1e-2,
        use_scheduler: bool = True,  
        model_params: dict = None,
        epochs: int = 100 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.epochs = epochs

        select_model = getattr(import_module("src.models"), model_name)
        self._model = select_model(model_params)

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
        
        VAL_AMP = False
        if VAL_AMP:
            with torch.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
        
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Apply sigmoid to logits to get probabilities for binary classification
        probs = torch.sigmoid(logits)

        # Binarize the probabilities (threshold at 0.5)
        y_pred = (probs > 0.5).float()

        # Since y is already binary (0 or 1), no need for further processing
        # Calculate Generalized Dice Score
        dice = compute_generalized_dice(y_pred, y)
        dice = dice.mean() if len(dice) > 0 else dice

        # Log the results
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        if self.use_scheduler:
            # Add lr scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            configuration["lr_scheduler"] = scheduler

        return configuration
    