"""Checkpoint compatibility and public inference contracts, without downloads."""

import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from ml_collections import ConfigDict
import pytest
import torch

from seisLM import encode, load_pretrained, save_pretrained
from seisLM.model.foundation.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining


@pytest.fixture
def model():
  from importlib.resources import files

  config = json.loads(files("seisLM.configs").joinpath(
    "pretrain/pretrain_config_std_norm_single_ax_8_datasets_32bit_scaleup_samp_false.json"
  ).read_text())["model_config"]
  config.update(
    conv_dim=[8, 8], hidden_size=8, intermediate_size=16,
    num_hidden_layers=1, num_attention_heads=2,
    num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
    codevector_dim=8, proj_codevector_dim=8, num_codevectors_per_group=4,
  )
  torch.manual_seed(42)
  return MultiDimWav2Vec2ForPreTraining(ConfigDict(config)).eval()


def test_bundle_roundtrip_preserves_all_weights_and_outputs(model, tmp_path):
  inputs = torch.randn(2, 3, 65)
  reference = encode(model, inputs)
  save_pretrained(model, tmp_path)
  restored = load_pretrained(tmp_path)
  assert not restored.training
  assert model.state_dict().keys() == restored.state_dict().keys()
  for name, value in model.state_dict().items():
    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
  torch.testing.assert_close(encode(restored, inputs), reference, rtol=0, atol=0)


def test_bundle_paths_expand_tilde_in_output_and_provenance(model, tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  checkpoint = tmp_path / "source.ckpt"
  checkpoint.write_bytes(b"checkpoint provenance")
  directory = tmp_path / "bundle"
  user_directory = "~/" + os.path.relpath(directory, Path.home())
  user_checkpoint = "~/" + os.path.relpath(checkpoint, Path.home())
  saved = save_pretrained(model, user_directory, source_checkpoint=user_checkpoint)
  assert saved.resolve() == directory
  metadata = json.loads((directory / "config.json").read_text())
  assert metadata["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  restored = load_pretrained(user_directory)
  for name, value in model.state_dict().items():
    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
  assert not (tmp_path / "~").exists()


def test_legacy_requires_explicit_trust_and_loads_embedded_config(model, tmp_path):
  checkpoint = tmp_path / "original.ckpt"
  torch.save({
    "hyper_parameters": {"config": ConfigDict({"model_config": model.config.to_dict()})},
    "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
  }, checkpoint)
  with pytest.raises(ValueError, match="trust_legacy_checkpoint"):
    load_pretrained(checkpoint)
  restored = load_pretrained(checkpoint, trust_legacy_checkpoint=True)
  assert restored.config.hidden_size == 8
  for name, value in model.state_dict().items():
    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_rejects_incompatible_bundle_weights(model, tmp_path):
  save_pretrained(model, tmp_path)
  config = json.loads((tmp_path / "config.json").read_text())
  config["model_config"]["num_hidden_layers"] = 2
  (tmp_path / "config.json").write_text(json.dumps(config))
  with pytest.raises(RuntimeError, match="Missing key"):
    load_pretrained(tmp_path)


def test_hub_loading_resolves_one_snapshot(model, tmp_path):
  save_pretrained(model, tmp_path)
  with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download:
    restored = load_pretrained("example/seislm", revision="fixed-revision", local_files_only=True)
  assert not restored.training
  assert download.call_args.kwargs["revision"] == "fixed-revision"
  assert download.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("inputs", [
  torch.ones(3, 65), torch.ones(1, 2, 65), torch.ones(1, 3, 2),
  torch.ones(1, 3, 65, dtype=torch.float64), torch.full((1, 3, 65), float("nan")),
  torch.empty(0, 3, 65),
])
def test_encode_rejects_invalid_input(model, inputs):
  with pytest.raises(ValueError):
    encode(model, inputs)


def test_local_bundle_does_not_need_hub_or_training_imports(model, tmp_path):
  save_pretrained(model, tmp_path)
  # In-process import blocking also guards against future accidental imports.
  blocked = {name: None for name in ("lightning", "seisbench", "transformers", "torchtune", "huggingface_hub")}
  with patch.dict(sys.modules, blocked):
    restored = load_pretrained(tmp_path)
    assert encode(restored, torch.ones(1, 3, 65)).shape == (1, 15, 8)


def test_missing_local_checkpoint_is_not_treated_as_hub_id(tmp_path):
  with pytest.raises(FileNotFoundError):
    load_pretrained(tmp_path / "missing.ckpt")


@pytest.mark.parametrize("stable_layer_norm", [False, True])
@pytest.mark.skipif(
  sys.version_info >= (3, 13),
  reason="The pinned 2024 Transformers reference requires Python 3.11–3.12; use saved-checkpoint regression on 3.13.",
)
def test_encoder_matches_reference_without_downloads(stable_layer_norm):
  """Exercise both normalization layouts against the pinned Wav2Vec2 reference."""
  from transformers import Wav2Vec2Config, Wav2Vec2Model as ReferenceEncoder
  from seisLM.model.foundation.multidim_wav2vec2 import Wav2Vec2Model

  config = Wav2Vec2Config(
    hidden_size=8, intermediate_size=16, num_hidden_layers=1,
    num_attention_heads=2, conv_dim=(8, 8), conv_kernel=(3, 3),
    conv_stride=(2, 2), conv_bias=True, feat_extract_norm="layer",
    num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
    hidden_dropout=0, attention_dropout=0, activation_dropout=0,
    feat_proj_dropout=0, layerdrop=0, mask_time_prob=0,
    do_stable_layer_norm=stable_layer_norm,
  )
  config.input_dim = 1
  config.use_rms_norm = False
  config.rotary_pos_embed = False
  config.conv_embed = True
  reference = ReferenceEncoder(config).eval()
  candidate = Wav2Vec2Model(config).eval()
  candidate.load_state_dict(reference.state_dict(), strict=True)
  inputs = torch.randn(2, 65)
  with torch.inference_mode():
    expected = reference(inputs)
    actual = candidate(inputs)
  torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=1e-5, atol=1e-6)
  torch.testing.assert_close(actual.extract_features, expected.extract_features, rtol=1e-5, atol=1e-6)
