import lightning as L
import torch
from monai.losses import DiceLoss
from monai.networks.nets import *
from monai.networks.layers import *
from monai.metrics import compute_generalized_dice
from monai.inferers import SimpleInferer, SlidingWindowInferer

class BrianModel(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-2,
        use_scheduler: bool = True,    
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        ## TODO CHANGE IT TO SELECT MODEL
        self._model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )

        self.criterion=DiceLoss(to_onehot_y=True, softmax=True)

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
        loss = self.criterion(logits, y)
        y_pred = self._inference(x)
        dice = compute_generalized_dice(y_pred, y)
        dice = dice.mean() if len(dice) > 0 else dice
        self.log("val_loss", loss, on_step = True, on_epoch = True, prob_bar = True)
        self.log("val_dice", dice, on_step = True, on_epoch = True, prob_bar = True)

    def configure_optimizers(self):
        optimizer = torch.optim.adamw(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        if self.use_scheduler:
            # Add lr scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)
            configuration["lr_scheduler"] = scheduler

        return configuration
    