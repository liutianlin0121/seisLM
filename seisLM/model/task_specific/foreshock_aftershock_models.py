""" Models for the foreshock-aftershock classification task. """
import torch.nn as nn

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

class SeqClassifier(nn.Module):
  def __init__(
    self,
    in_channels,
    num_classes,
    num_layers=3,
    initial_filters=16,
    kernel_size=3
  ):
    super(SeqClassifier, self).__init__()
    layers = []
    for i in range(num_layers):
      out_channels = initial_filters * (2 ** i)
      layers.append(DoubleConvBlock(in_channels, out_channels, kernel_size))
      in_channels = out_channels

    self.conv_encoder = nn.Sequential(*layers)
    self.global_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(out_channels, num_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.conv_encoder(x)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = self.softmax(x)
    return x
