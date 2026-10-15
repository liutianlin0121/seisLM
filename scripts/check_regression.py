"""Capture original checkpoint outputs and compare the practitioner loader.

Run this script with the environment and source checkout being evaluated.
References are local tensor artifacts; capture refuses to replace existing ones.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


SEED = 42
FIELDS = (
  "last_hidden_state", "extract_features", "projected_states",
  "projected_quantized_states", "codevector_perplexity", "loss",
  "contrastive_loss", "diversity_loss",
)


def sha256(path):
  with Path(path).open("rb") as handle:
    return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate(model, inputs, device):
  from seisLM.model.foundation.mask_utils import (
    compute_mask_indices, get_feat_extract_output_lengths,
    sample_negative_indices,
  )

  if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU.")
  model = model.to(device).eval()
  inputs = inputs.to(device)
  sequence_length = int(get_feat_extract_output_lengths(
    model.config, inputs.shape[-1],
  ))
  np.random.seed(SEED)
  masks = compute_mask_indices(
    (inputs.shape[0], sequence_length), mask_prob=0.2,
    mask_length=10, min_masks=2,
  )
  np.random.seed(SEED)
  negatives = sample_negative_indices(
    (inputs.shape[0], sequence_length), num_negatives=10,
    mask_time_indices=masks,
  )
  torch.manual_seed(SEED)
  with torch.inference_mode():
    encoder = model.wav2vec2(inputs, output_hidden_states=True)
    unmasked = model(inputs)
    masked = model(
      inputs, mask_time_indices=torch.as_tensor(masks, device=device),
      sampled_negative_indices=torch.as_tensor(negatives, device=device),
    )
  values = {
    "last_hidden_state": encoder.last_hidden_state,
    "extract_features": encoder.extract_features,
    "projected_states": unmasked.projected_states,
    "projected_quantized_states": unmasked.projected_quantized_states,
    "codevector_perplexity": unmasked.codevector_perplexity,
    "loss": masked.loss,
    "contrastive_loss": masked.contrastive_loss,
    "diversity_loss": masked.diversity_loss,
  }
  return {key: value.detach().cpu() for key, value in values.items()}


def json_write(path, data):
  Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def provenance(model):
  import inspect

  source = Path(inspect.getfile(type(model))).resolve()
  try:
    commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=source.parent,
      stderr=subprocess.DEVNULL, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=no"],
      cwd=source.parent, stderr=subprocess.DEVNULL, text=True,
    ).strip())
  except (OSError, subprocess.CalledProcessError):
    commit = None
    dirty = None
  return {
    "source_commit": commit, "tracked_source_changes": dirty,
    "source_module": str(source),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": sys.version.split()[0],
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
  }


def capture(args):
  from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2

  if not args.trust_legacy_checkpoint:
    raise ValueError("Capture requires --trust-legacy-checkpoint for the original pickle loader.")
  reference = args.reference_dir
  reference.mkdir(parents=True, exist_ok=True)
  output = reference / f"{args.name}-outputs.pt"
  config_path = reference / f"{args.name}-config.json"
  metadata_path = reference / f"{args.name}-capture.json"
  for path in (output, config_path, metadata_path):
    if path.exists():
      raise FileExistsError(f"Reference already exists: {path}; choose a new reference directory.")
  fixture = reference / "input.pt"
  if not fixture.exists():
    if args.input is None:
      raise ValueError("Supply --input with a saved normalized waveform tensor for the first capture.")
    inputs = torch.load(args.input, map_location="cpu", weights_only=True)
  else:
    if args.input is not None and sha256(args.input) != sha256(fixture):
      raise ValueError("--input differs from the reference directory's existing input.pt.")
    inputs = torch.load(fixture, map_location="cpu", weights_only=True)
  validate_inputs(inputs)
  model = LitMultiDimWav2Vec2.load_from_checkpoint(
    args.checkpoint, map_location="cpu", weights_only=False,
  ).model.eval()
  outputs = {device: evaluate(model, inputs, device) for device in args.device}
  metadata = provenance(model)
  metadata.update({
    "checkpoint_sha256": sha256(args.checkpoint),
    "input_shape": list(inputs.shape), "devices": args.device,
    "mask_seed": SEED, "mask_probability": 0.2, "mask_length": 10,
    "minimum_masks": 2, "negatives_per_position": 10,
    "fixture_description": args.fixture_description,
  })
  if not fixture.exists():
    torch.save(inputs, fixture)
  metadata["input_sha256"] = sha256(fixture)
  config = model.config.to_dict()
  torch.save(outputs, output)
  json_write(config_path, config)
  json_write(metadata_path, metadata)
  print(json.dumps(metadata, indent=2))
  return 0


def validate_inputs(inputs):
  if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or
      inputs.shape[1] != 3 or inputs.dtype != torch.float32 or
      not torch.isfinite(inputs).all()):
    raise ValueError("Fixture must be a finite float32 tensor [batch, 3, samples].")


def compare_tensor(actual, expected, atol, rtol):
  result = {"shape": list(actual.shape), "reference_shape": list(expected.shape)}
  if actual.shape != expected.shape:
    return {**result, "passed": False, "reason": "shape mismatch"}
  if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
    return {**result, "passed": False, "reason": "nonfinite values"}
  delta = (actual.double() - expected.double()).abs()
  denominator = expected.double().abs().clamp_min(torch.finfo(torch.float64).eps)
  return {
    **result, "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
    "exactly_equal": bool(torch.equal(actual, expected)),
    "max_absolute_error": delta.max().item(),
    "max_relative_error": (delta / denominator).max().item(),
  }


def compare(args):
  from seisLM import encode, load_pretrained

  reference = args.reference_dir
  inputs = torch.load(reference / "input.pt", map_location="cpu", weights_only=True)
  validate_inputs(inputs)
  expected = torch.load(
    reference / f"{args.name}-outputs.pt", map_location="cpu", weights_only=True,
  )
  source = args.bundle if args.bundle else args.checkpoint
  model = load_pretrained(
    source, device="cpu", trust_legacy_checkpoint=args.trust_legacy_checkpoint,
  )
  report = {
    "model": args.name, "source": str(source),
    "reference_dir": str(reference.resolve()),
    "input_sha256": sha256(reference / "input.pt"),
    "atol": args.atol, "rtol": args.rtol,
    "environment": provenance(model), "checks": {},
  }
  config_path = reference / f"{args.name}-config.json"
  if config_path.exists():
    report["config_matches"] = model.config.to_dict() == json.loads(config_path.read_text())
  for device in args.device:
    if device not in expected:
      raise ValueError(f"No {device} reference for {args.name}; capture it before comparison.")
    actual = evaluate(model, inputs, device)
    checks = {}
    for field in FIELDS:
      if field not in expected[device]:
        raise ValueError(f"Reference is missing {device}/{field}.")
      checks[field] = compare_tensor(actual[field], expected[device][field], args.atol, args.rtol)
    checks["public_encode"] = compare_tensor(
      encode(model, inputs).cpu(), expected[device]["last_hidden_state"],
      args.atol, args.rtol,
    )
    report["checks"][device] = checks
  report["passed"] = report.get("config_matches", True) and all(
    check["passed"] for checks in report["checks"].values() for check in checks.values()
  )
  if args.report:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    json_write(args.report, report)
  print(json.dumps(report, indent=2))
  return 0 if report["passed"] else 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  for command in ("capture", "compare"):
    sub = commands.add_parser(command)
    sub.add_argument("--name", choices=("base", "large"), required=True)
    sub.add_argument("--reference-dir", type=Path, required=True)
    sub.add_argument("--checkpoint", type=Path)
    sub.add_argument("--device", choices=("cpu", "cuda"), action="append")
    sub.add_argument("--source-root", type=Path, help="Import SeisLM from this checkout.")
    sub.add_argument("--trust-legacy-checkpoint", action="store_true")
    if command == "capture":
      sub.add_argument("--input", type=Path, help="Already normalized, saved tensor; reused verbatim.")
      sub.add_argument("--fixture-description", default="User-supplied normalized ZNE waveforms at 100 Hz.")
    else:
      sub.add_argument("--bundle", type=Path)
      sub.add_argument("--atol", type=float, default=1e-6)
      sub.add_argument("--rtol", type=float, default=1e-5)
      sub.add_argument("--report", type=Path)
  args = parser.parse_args()
  args.device = list(dict.fromkeys(args.device or ["cpu"]))
  if args.source_root:
    sys.path.insert(0, str(args.source_root.resolve()))
  if args.command == "capture" and args.checkpoint is None:
    parser.error("capture requires --checkpoint")
  if args.command == "compare":
    if bool(args.checkpoint) == bool(args.bundle):
      parser.error("compare requires exactly one of --checkpoint and --bundle")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
      parser.error("tolerances must be finite and nonnegative")
  try:
    return capture(args) if args.command == "capture" else compare(args)
  except (ValueError, RuntimeError, OSError) as exc:
    print(f"Regression check failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
