from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
from torch.optim import Optimizer

class BaseModelUnfreeze(BaseFinetuning):
  def __init__(self, unfreeze_at_epoch: int = 10):
    super().__init__()
    self._unfreeze_at_epoch = unfreeze_at_epoch

  def freeze_before_training(self, pl_module: LightningModule) -> None:
    # freeze any module you want
    # Here, we are freezing `feature_extractor`
    self.freeze(pl_module.model.wav2vec2)

  def finetune_function(
    self,
    pl_module: LightningModule,
    current_epoch: int,
    optimizer: Optimizer,
    ) -> None:
    # When `current_epoch` is 10, feature_extractor will start training.
    if current_epoch == self._unfreeze_at_epoch:
        self.unfreeze_and_add_param_group(
            modules=pl_module.model.wav2vec2,
            optimizer=optimizer,
            train_bn=True,
        )
