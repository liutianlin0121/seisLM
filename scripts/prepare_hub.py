"""Prepare a checksum-verified public encoder release locally; never uploads."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from seisLM import MODEL_CATALOG, load_pretrained, save_pretrained


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--model", choices=MODEL_CATALOG, required=True)
  parser.add_argument("--trust-legacy-checkpoint", action="store_true")
  parser.add_argument("--wheel", type=Path, help="Include the reviewed package wheel for standalone installation.")
  args = parser.parse_args()
  entry = MODEL_CATALOG[args.model]
  with args.checkpoint.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
  if digest != entry["legacy_sha256"]:
    parser.error("Checkpoint checksum does not match the original public release.")
  if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
    parser.error("Output directory must be empty to avoid mixing release artifacts.")
  if args.wheel is not None and (not args.wheel.is_file() or args.wheel.suffix != ".whl"):
    parser.error("--wheel must point to an existing built Python wheel.")
  model = load_pretrained(args.checkpoint, trust_legacy_checkpoint=args.trust_legacy_checkpoint)
  save_pretrained(model, args.output, source_checkpoint=args.checkpoint)
  params = sum(p.numel() for p in model.parameters())
  installation = "Install the compatible seisLM package from the linked code repository."
  if args.wheel is not None:
    shutil.copyfile(args.wheel, args.output / args.wheel.name)
    installation = f'''A compatible Python wheel is included so this release can be used before
the source changes are merged or published to PyPI. Install with Python 3.11–3.13:

```bash
python -m pip install "https://huggingface.co/{entry['repo_id']}/resolve/main/{args.wheel.name}"
```

For reproducible installation, replace `main` in the wheel URL with the reviewed
model-repository commit. For GPU-specific PyTorch installation, use the
[PyTorch selector](https://pytorch.org/get-started/locally/).'''
  card = f"""---
license: mit
library_name: seislm
pipeline_tag: feature-extraction
tags:
  - seismology
  - seismic-waveforms
  - self-supervised-learning
  - arxiv:2410.15765
---

# SeisLM {args.model}

Pretrained seismic waveform encoder from [SeisLM: a Foundation Model for
Seismic Waveforms](https://arxiv.org/abs/2410.15765).
Code and installation: https://github.com/liutianlin0121/seisLM

This is an encoder with the original self-supervised pretraining heads. It
does **not** include a trained phase-picking or earthquake-detection head.
Downstream adaptation is needed for those tasks. It is not a Transformers
AutoModel artifact or a hosted inference endpoint.

## Architecture and provenance

- Parameters (including pretraining heads): {params:,}
- Transformer layers: {model.config.num_hidden_layers}
- Contextual feature dimension: {model.config.hidden_size}
- Original checkpoint SHA-256: `{digest}`
- Export: unchanged state tensors in safetensors plus the architecture
  configuration embedded in the original checkpoint. Optimizer state omitted.
- Training sources recorded in the original checkpoint: ETHZ, GEOFON, STEAD,
  MLAAPDE, InstanceCounts, Iquique, PNW, OBST2024.

## Input and output

Inputs: float32 tensors `[batch, 3, samples]`, components **ZNE**, **100 Hz**.
Reference window: 3001 samples. Demean each component, then divide by its
population standard deviation plus `1e-10` (replace zero standard deviation
by 1). No automatic resampling, filtering, or component filling is performed.
See `preprocessing.json` for the contract and missing-component caveat.

Contextual features have shape `[batch, reduced_time, {model.config.hidden_size}]`.
For 3001 input samples, reduced_time is 749. These are features, not P/S picks.

## Loading

{installation}

```python
import torch
from seisLM import load_pretrained, encode

model = load_pretrained("{entry['repo_id']}")
# Synthetic installation smoke example, not a scientific evaluation:
x = torch.randn(1, 3, 3001, generator=torch.Generator().manual_seed(2024))
x = x - x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, correction=0, keepdim=True)
x = x / (torch.where(std == 0, torch.ones_like(std), std) + 1e-10)
features = encode(model, x)
print(features.shape)
```

Set `revision` to a published model-repository commit for repeatable downloads.
A local bundle directory also works offline.

## Refactoring regression check

The exported weights were compared with the original Lightning-loaded checkpoint
on fixed complete/noise/zero-channel inputs using CPU and NVIDIA GB10 CUDA,
PyTorch 2.14.1+cu130, Python 3.12.3. All eight checked fields (encoder features,
convolutional features, projected/quantized features, codebook perplexity, and
three masked-loss values) matched exactly, with maximum absolute error 0.
This is numerical parity on a regression fixture, not a downstream benchmark.
See the source repository's regression documentation for the protocol.

## Evaluation and limitations

The original paper evaluates fine-tuned task models, not this bare encoder.
The paper's downstream scores cannot be inferred from successful loading or
numerical parity of this artifact. Local adaptation and held-out evaluation
are needed before relying on predictions for a new region or instrument.

## License

These pretrained weights are released under the MIT license, as confirmed by
the author. See LICENSE. The code is also distributed under MIT.

## Citation

Tianlin Liu et al. (2024). SeisLM: a Foundation Model for Seismic Waveforms.
arXiv:2410.15765. https://doi.org/10.48550/arXiv.2410.15765
"""
  (args.output / "README.md").write_text(card, encoding="utf-8")
  license_path = Path(__file__).resolve().parents[1] / "LICENSE"
  (args.output / "LICENSE").write_text(license_path.read_text(), encoding="utf-8")
  manifest = {"model": args.model, "repo_id": entry["repo_id"], "files": {}}
  for path in sorted(args.output.iterdir()):
    if path.is_file():
      with path.open("rb") as handle:
        manifest["files"][path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
  (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"Prepared {args.output} for {entry['repo_id']} ({params:,} parameters). Nothing uploaded.")


if __name__ == "__main__":
  main()
