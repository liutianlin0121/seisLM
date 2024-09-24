import unittest

import ml_collections
import torch

# Assuming the classes have been imported from your module
from seisLM.model.foundation.conv_encoder import (
  Wav2Vec2FeatureEncoder,
  Wav2Vec2GroupNormConvLayer,
  Wav2Vec2LayerNormConvLayer,
  Wav2Vec2NoLayerNormConvLayer,
)


class TestFeatureEncodersForwardPass(unittest.TestCase):
  def setUp(self) -> None:
    # Common configuration for all tests
    self.batch_size = 2
    self.seq_length = 16000  # Example sequence length for audio data
    self.input_dim = 1  # Mono audio
    self.config = ml_collections.ConfigDict()
    self.config.input_dim = self.input_dim
    self.config.conv_dim = [32, 64]
    self.config.conv_kernel = [5, 3]
    self.config.conv_stride = [2, 2]
    self.config.conv_bias = False
    self.config.use_rms_norm = False
    self.config.num_feat_extract_layers = len(self.config.conv_dim)
    self.config.feat_extract_norm = "layer"

  def test_no_layer_norm_conv_layer_forward(self) -> None:
    layer_id = 0
    conv_layer = Wav2Vec2NoLayerNormConvLayer(self.config, layer_id)
    input_tensor = torch.randn(self.batch_size, self.input_dim, self.seq_length)
    output = conv_layer(input_tensor)
    expected_output_dim = self.config.conv_dim[layer_id]
    expected_length = (
      (self.seq_length - self.config.conv_kernel[layer_id])
      // self.config.conv_stride[layer_id]
    ) + 1
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, expected_length)
    )

  def test_layer_norm_conv_layer_forward(self) -> None:
    layer_id = 0
    conv_layer = Wav2Vec2LayerNormConvLayer(self.config, layer_id)
    input_tensor = torch.randn(self.batch_size, self.input_dim, self.seq_length)
    output = conv_layer(input_tensor)
    expected_output_dim = self.config.conv_dim[layer_id]
    expected_length = (
      (self.seq_length - self.config.conv_kernel[layer_id])
      // self.config.conv_stride[layer_id]
    ) + 1
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, expected_length)
    )

  def test_group_norm_conv_layer_forward(self) -> None:
    layer_id = 0
    conv_layer = Wav2Vec2GroupNormConvLayer(self.config, layer_id)
    input_tensor = torch.randn(self.batch_size, self.input_dim, self.seq_length)
    output = conv_layer(input_tensor)
    expected_output_dim = self.config.conv_dim[layer_id]
    expected_length = (
      (self.seq_length - self.config.conv_kernel[layer_id])
      // self.config.conv_stride[layer_id]
    ) + 1
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, expected_length)
    )

  def test_feature_encoder_forward_with_layer_norm(self) -> None:
    encoder = Wav2Vec2FeatureEncoder(self.config)
    input_tensor = torch.randn(self.batch_size, self.seq_length)
    output = encoder(input_tensor)
    # Calculate expected output length after all convolutional layers
    length = self.seq_length
    for kernel_size, stride in zip(
      self.config.conv_kernel, self.config.conv_stride
    ):
      length = ((length - kernel_size) // stride) + 1
    expected_output_dim = self.config.conv_dim[-1]
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, length)
    )

  def test_feature_encoder_forward_with_group_norm(self) -> None:
    self.config.feat_extract_norm = "group"
    encoder = Wav2Vec2FeatureEncoder(self.config)
    input_tensor = torch.randn(self.batch_size, self.seq_length)
    output = encoder(input_tensor)
    # Calculate expected output length after all convolutional layers
    length = self.seq_length
    for kernel_size, stride in zip(
      self.config.conv_kernel, self.config.conv_stride
    ):
      length = ((length - kernel_size) // stride) + 1
    expected_output_dim = self.config.conv_dim[-1]
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, length)
    )

  def test_feature_encoder_forward_with_3d_input(self) -> None:
    encoder = Wav2Vec2FeatureEncoder(self.config)
    input_tensor = torch.randn(self.batch_size, self.input_dim, self.seq_length)
    output = encoder(input_tensor)
    # Calculate expected output length after all convolutional layers
    length = self.seq_length
    for kernel_size, stride in zip(
      self.config.conv_kernel, self.config.conv_stride
    ):
      length = ((length - kernel_size) // stride) + 1
    expected_output_dim = self.config.conv_dim[-1]
    self.assertEqual(
      output.shape, (self.batch_size, expected_output_dim, length)
    )

  def test_forward_pass_without_errors(self) -> None:
    # Test that the forward pass runs without runtime errors
    encoder = Wav2Vec2FeatureEncoder(self.config)
    input_tensor = torch.randn(self.batch_size, self.seq_length)
    try:
      _ = encoder(input_tensor)
    except Exception as e:
      self.fail(f"Forward pass raised an exception: {e}")


if __name__ == "__main__":
  unittest.main()
