"""Extract features with an existing pretrained SeisLM checkpoint.

Run from an installed environment, for example:
  uv run --locked python examples/quickstart.py --checkpoint PATH
Synthetic inputs demonstrate execution, not scientific performance.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from seisLM import MODEL_CATALOG, encode, load_pretrained


def normalize(waveforms: torch.Tensor) -> torch.Tensor:
  """Apply the reference per-component demeaning/std normalization."""
  waveforms = waveforms - waveforms.mean(dim=-1, keepdim=True)
  std = waveforms.std(dim=-1, correction=0, keepdim=True)
  std = torch.where(std == 0, torch.ones_like(std), std)
  return waveforms / (std + 1e-10)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint", required=True,
    help="Local bundle, official legacy .ckpt, or published Hub repository",
  )
  parser.add_argument(
    "--waveform", type=Path,
    help="100 Hz ZNE float32 .npy array, [3,T] or [B,3,T]",
  )
  parser.add_argument(
    "--device", default="auto",
    help="auto (CUDA if available, else CPU), cpu, or cuda",
  )
  parser.add_argument("--revision", help="Published Hub commit SHA")
  parser.add_argument("--cache-dir", type=Path)
  parser.add_argument("--local-files-only", action="store_true")
  parser.add_argument(
    "--trust-legacy-checkpoint", action="store_true",
    help="Allow checksum-verified official Python-serialized checkpoint",
  )
  args = parser.parse_args()

  checkpoint = Path(args.checkpoint).expanduser()
  if checkpoint.is_file():
    if not args.trust_legacy_checkpoint:
      parser.error(
        "Legacy checkpoint loading requires --trust-legacy-checkpoint; "
        "export a portable bundle to avoid Python serialization."
      )
    with checkpoint.open("rb") as handle:
      checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    official = {item["legacy_sha256"] for item in MODEL_CATALOG.values()}
    if checksum not in official:
      parser.error(
        "Checkpoint SHA-256 does not match an official base/large release. "
        "This quickstart only trusts checksum-verified official checkpoints."
      )
    print(f"Verified original checkpoint SHA-256: {checksum}")

  device = args.device
  if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
  model = load_pretrained(
    args.checkpoint,
    revision=args.revision,
    device=device,
    cache_dir=args.cache_dir,
    local_files_only=args.local_files_only,
    trust_legacy_checkpoint=args.trust_legacy_checkpoint,
  )

  if args.waveform:
    array = np.load(args.waveform, allow_pickle=False)
    if array.dtype != np.float32:
      parser.error("Waveform array must be float32.")
    waveforms = torch.from_numpy(array)
    if waveforms.ndim == 2:
      waveforms = waveforms.unsqueeze(0)
    if waveforms.ndim != 3 or waveforms.shape[1] != 3:
      parser.error("Waveform shape must be [3, samples] or [batch, 3, samples].")
  else:
    # Fixed CPU generator makes the input independent of model/device.
    generator = torch.Generator().manual_seed(2024)
    waveforms = torch.randn(1, 3, 3001, generator=generator)
    print(
      "Input: synthetic installation smoke example, 100 Hz ZNE; "
      "not a benchmark."
    )

  features = encode(model, normalize(waveforms))
  print(f"Device: {next(model.parameters()).device}")
  print(f"Input shape: {tuple(waveforms.shape)}")
  print(f"Feature shape: {tuple(features.shape)}")
  print(f"Finite features: {bool(torch.isfinite(features).all())}")


if __name__ == "__main__":
  main()
