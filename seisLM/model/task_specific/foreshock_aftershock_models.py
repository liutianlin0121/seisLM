""" Models for the foreshock-aftershock classification task. """
import torch
import torch.nn as nn
import torchmetrics
import lightning as L

class DoubleConvBlock(nn.Module):
  """Two conv layers with batchnorm and ReLU activation, like in a 1d U-Net."""
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout_rate: float = 0.1
    ):
    super(DoubleConvBlock, self).__init__()

    conv_shared_kwargs = {
      'kernel_size': kernel_size,
      'out_channels': out_channels,
      'padding': 'valid',
      'bias': False, # Because batchnorm follows the conv layer.
    }

    self.double_conv = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, stride=1, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
      nn.Conv1d(in_channels=out_channels, stride=2, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
    )

  def forward(self, x):
    x = self.double_conv(x)
    return x


class Conv1DShockClassifier(nn.Module):
  """A simple 1D conv classifier for foreshock-aftershock classification."""
  def __init__(
    self,
    in_channels: int,
    num_classes: int,
    num_layers: int = 3,
    initial_filters: int = 16,
    kernel_size: int = 3,
    dropout_rate: float = 0.1
  ):
    super(Conv1DShockClassifier, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers

    layers = []
    for i in range(num_layers):
      out_channels = initial_filters * (2 ** i)
      layers.append(
        DoubleConvBlock(in_channels, out_channels, kernel_size, dropout_rate)
      )
      in_channels = out_channels

    self.conv_encoder = nn.Sequential(*layers)
    self.global_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(out_channels, num_classes)

  def forward(self, x):
    x = self.conv_encoder(x)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class Conv1DShockClassifierLit(L.LightningModule):
  """ A LightningModule for the Conv1DShockClassifier model. """
  def __init__(
    self,
    model_config,
    max_train_steps: int,
    ):
    super().__init__()
    self.save_hyperparameters()
    self.model_config = model_config
    self.max_train_steps = max_train_steps

    self.model = Conv1DShockClassifier(
      in_channels=model_config.in_channels,
      num_classes=model_config.num_classes,
      num_layers=model_config.num_layers,
      initial_filters=model_config.initial_filters,
      kernel_size=model_config.kernel_size
    )

    num_classes = self.model.num_classes
    self.train_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=num_classes
    )
    self.val_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=num_classes
    )
    self.test_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=num_classes
    )

  def forward(self, waveforms):
    logits = self.model(waveforms)
    return logits

  def training_step(self, batch, batch_idx):
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.train_acc(predicted_labels, labels)

    self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True)
    self.log("train/acc", self.train_acc, sync_dist=True, prog_bar=True)

    return loss  # this is passed to the optimizer for training

  def validation_step(self, batch, batch_idx):
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.val_acc(predicted_labels, labels)
    self.log("val/loss", loss, sync_dist=True, prog_bar=True)
    self.log("val/acc", self.val_acc, sync_dist=True, prog_bar=True)


  def test_step(self, batch, batch_idx):
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.test_acc(predicted_labels, labels)
    self.log("test/loss", loss, sync_dist=True, prog_bar=True)
    self.log("test/acc", self.test_acc, sync_dist=True, prog_bar=True)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.trainer.model.parameters(),
        lr=self.model_config.learning_rate,
        weight_decay=self.model_config.weight_decay,
    )
    tmax = int(self.max_train_steps // self.trainer.num_devices)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=tmax
    )
    sched_config = {
        'scheduler': scheduler,
        'interval': "step",
        'frequency': 1,
    }

    return {"optimizer": optimizer, "lr_scheduler": sched_config}
