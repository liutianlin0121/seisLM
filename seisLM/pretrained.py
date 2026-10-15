"""Portable loading of the existing pretrained SeisLM encoders.

This module does not import Lightning, SeisBench, Transformers, or torchtune
for the published LayerNorm models. It does not implement a phase-picking head.
"""

import hashlib
import json
from pathlib import Path

from ml_collections import ConfigDict
import torch

from seisLM.model.foundation.multidim_wav2vec2 import (
  MultiDimWav2Vec2ForPreTraining,
)


MODEL_CATALOG = {
  "base": {
    "repo_id": "ltlltl2006/seislm-base",
    "parameters": 11359568,
    "hidden_size": 240,
    "layers": 6,
    "legacy_file_id": "12L2Js1DIXCIxacQJ4K0hRKJ6qHFrQ7K-",
    "legacy_sha256": "6e97c53c562e9ba6bc96df85d8d208b89ff84e80a25468825f29bb5bace5eacb",
  },
  "large": {
    "repo_id": "ltlltl2006/seislm-large",
    "parameters": 90683648,
    "hidden_size": 768,
    "layers": 12,
    "legacy_file_id": "1LPUCltHsle4OqDPGEtcHPdoRnLppY-KH",
    "legacy_sha256": "db44b1a46512de2cdf5c310c7ccfdac0ef04931cc89b7768fad41c978266df4b",
  },
}

PREPROCESSING = {
  "sampling_rate_hz": 100,
  "component_order": "ZNE",
  "tensor_layout": "batch, component, sample",
  "dtype": "float32",
  "reference_window_samples": 3001,
  "normalization": "per-component demean, divide by population std + 1e-10",
  "zero_std": "replace zero std by 1 before division",
  "missing_components": (
    "Legacy training randomly copies a present component into missing channels. "
    "Supply complete components for deterministic inference; no filling is "
    "performed by the loader or encode()."
  ),
}


def _legacy_model(path: Path, trust_legacy_checkpoint: bool):
  if not trust_legacy_checkpoint:
    raise ValueError(
      "Legacy Lightning checkpoints contain pickled Python objects. Use a "
      "portable bundle, or set trust_legacy_checkpoint=True only for a trusted "
      "checkpoint (for example the author's checksum-verified release)."
    )
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  try:
    config = checkpoint["hyper_parameters"]["config"]
    if hasattr(config, "to_dict"):
      config = config.to_dict()
    config = ConfigDict(config["model_config"])
    state = checkpoint["state_dict"]
  except (KeyError, TypeError) as exc:
    raise ValueError(
      "Expected a pretrained LitMultiDimWav2Vec2 checkpoint. Downstream task "
      "checkpoints require their task-specific loader."
    ) from exc
  model = MultiDimWav2Vec2ForPreTraining(config)
  # Keep strict loading: a task head or incompatible architecture must fail.
  state = {key.removeprefix("model."): value for key, value in state.items()}
  model.load_state_dict(state, strict=True)
  return model


def load_pretrained(
  source: str | Path,
  *,
  revision: str | None = None,
  device: str | torch.device = "cpu",
  cache_dir: str | Path | None = None,
  local_files_only: bool = False,
  trust_legacy_checkpoint: bool = False,
) -> MultiDimWav2Vec2ForPreTraining:
  """Load an encoder/pretraining model in evaluation mode.

  ``source`` is a local bundle directory, a trusted legacy .ckpt file, a Hub
  repository ID, or the catalog name ``base``/``large``. Pin ``revision`` to a
  Hub commit for reproducibility. Normalization/resampling is the caller's
  responsibility. Local paths work without authentication or network access.
  """
  path = Path(source).expanduser()
  if path.is_file():
    model = _legacy_model(path, trust_legacy_checkpoint)
  else:
    if not path.is_dir():
      source_text = str(source)
      if (isinstance(source, Path) or source_text.startswith((".", "/", "~"))
          or source_text.endswith(".ckpt")):
        raise FileNotFoundError(f"Checkpoint or bundle does not exist: {path}")
      from huggingface_hub import snapshot_download

      repo_id = MODEL_CATALOG.get(source_text, {}).get("repo_id", source_text)
      # Resolve a single snapshot so config and weights use the same revision.
      path = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=["config.json", "model.safetensors", "preprocessing.json"],
      ))
    config_path = path / "config.json"
    weights_path = path / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
      raise ValueError(
        f"{path} must contain config.json and model.safetensors. "
        "Use scripts/prepare_hub.py to export a legacy pretrained checkpoint."
      )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if (metadata.get("format_version") != 1 or
        metadata.get("architecture") != "MultiDimWav2Vec2ForPreTraining"):
      raise ValueError("Unsupported SeisLM bundle format or architecture.")
    from safetensors.torch import load_file

    model = MultiDimWav2Vec2ForPreTraining(ConfigDict(metadata["model_config"]))
    model.load_state_dict(load_file(str(weights_path)), strict=True)
  return model.to(device).eval()


def save_pretrained(
  model: MultiDimWav2Vec2ForPreTraining,
  directory: str | Path,
  *,
  source_checkpoint: str | Path | None = None,
) -> Path:
  """Export unchanged weights plus JSON config, without training dependencies.

  This is an inference artifact; optimizer state is intentionally absent. Keep
  the original Lightning checkpoint to resume the original training workflow.
  """
  if not isinstance(model, MultiDimWav2Vec2ForPreTraining):
    raise TypeError("Only pretrained SeisLM encoder/pretraining models are supported.")
  from safetensors.torch import save_file

  target = Path(directory).expanduser()
  target.mkdir(parents=True, exist_ok=True)
  config = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
  metadata = {
    "format_version": 1,
    "architecture": "MultiDimWav2Vec2ForPreTraining",
    "model_config": config,
    "preprocessing": PREPROCESSING,
  }
  if source_checkpoint is not None:
    with Path(source_checkpoint).expanduser().open("rb") as handle:
      metadata["source_checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
  state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
  save_file(state, str(target / "model.safetensors"))
  (target / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
  (target / "preprocessing.json").write_text(json.dumps(PREPROCESSING, indent=2) + "\n", encoding="utf-8")
  return target


def encode(model: MultiDimWav2Vec2ForPreTraining, waveforms: torch.Tensor) -> torch.Tensor:
  """Extract contextual features [batch, reduced_time, hidden_size].

  Supply already normalized float32 waveforms [batch, 3, samples], at 100 Hz
  in ZNE order for released models. This function validates tensor structure;
  sampling rate, physical units, and channel orientation require caller metadata.
  It runs without gradients and requires evaluation mode for reproducibility.
  """
  if model.training:
    raise ValueError("Call model.eval() before extracting inference features.")
  if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 3:
    raise ValueError("Expected a torch.Tensor with shape [batch, component, sample].")
  if waveforms.shape[0] == 0 or waveforms.shape[1] != model.config.input_dim:
    raise ValueError(f"Expected a non-empty batch with {model.config.input_dim} components (ZNE).")
  length = waveforms.shape[-1]
  for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
    length = (length - kernel) // stride + 1
  if length < 1:
    raise ValueError("Waveform is too short for the convolutional encoder.")
  if waveforms.dtype != torch.float32 or not torch.isfinite(waveforms).all():
    raise ValueError("Waveforms must be finite float32 values.")
  parameter = next(model.parameters())
  with torch.inference_mode():
    return model.wav2vec2(waveforms.to(parameter.device)).last_hidden_state
