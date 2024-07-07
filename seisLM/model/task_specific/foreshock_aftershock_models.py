""" Models for the foreshock-aftershock classification task. """
import torch
import torch.nn as nn
import torchmetrics
import lightning as L

class DoubleConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size):
    super(DoubleConvBlock, self).__init__()

    conv_shared_kwargs = {
      'kernel_size': kernel_size,
      'out_channels': out_channels,
      'padding': 'valid',
      'bias': False, # Because batchnorm follows the conv layer.
    }

    self.double_conv = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, stride=2, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv1d(in_channels=out_channels, stride=1, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    x = self.double_conv(x)
    return x

class Conv1DShockClassifier(nn.Module):
  def __init__(
    self,
    in_channels,
    num_classes,
    num_layers=3,
    initial_filters=16,
    kernel_size=3
  ):
    super(Conv1DShockClassifier, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers

    layers = []
    for i in range(num_layers):
      out_channels = initial_filters * (2 ** i)
      layers.append(DoubleConvBlock(in_channels, out_channels, kernel_size))
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
  def __init__(
    self,
    in_channels,
    num_classes,
    num_layers,
    initial_filters,
    kernel_size,
    learning_rate
    ):
    super().__init__()
    self.save_hyperparameters()
    self.learning_rate = learning_rate
    self.model = Conv1DShockClassifier(
      in_channels=in_channels,
      num_classes=num_classes,
      num_layers=num_layers,
      initial_filters=initial_filters,
      kernel_size=kernel_size
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
    optimizer = torch.optim.Adam(
        self.trainer.model.parameters(), lr=self.learning_rate
    )
    return optimizer
