# Checking that modernization preserves model behavior

Two checks answer different questions: numerical comparisons against the
unmodified source protect this refactor, while a benchmark run would assess the
published downstream results. Passing the first does not establish the second.

## Numerical reference

Before changing the model code, both released pretrained checkpoints were loaded
through the original Lightning loader at commit
`ae1d27ac498ec7ae106e56eeaac386a0a5dd124f`. Reference tensors were captured on CPU
and the NVIDIA GB10 GPU in a DGX Spark. This baseline uses a working modern
environment; it does not reconstruct the original 2024 PyTorch environment.

The local `.artifacts/baseline/` directory holds the input tensor, architecture
configurations, CPU/CUDA outputs, checkpoint checksums, and environment manifest.
These large downloaded/generated artifacts are ignored by Git. Keep the
reference directory with a release's validation records; never regenerate it
using changed code to make a failing comparison pass.

The fixture has shape `[3, 3, 3001]` and dtype float32. It contains an ObsPy
bundled three-component recording padded to 3001 samples, seeded noise with a
zero component, and complete seeded noise. Channels use ZNE order, with
per-channel mean subtraction and division by population standard deviation plus
`1e-10` (zero standard deviations replaced by one). No stochastic missing-channel
filling is applied. The model receives these saved values verbatim.

Compare both the portable bundles and the trusted original checkpoints:

```bash
uv run --locked python scripts/check_regression.py compare \
  --name base --reference-dir .artifacts/baseline \
  --bundle .artifacts/hub-next/seislm-base --device cuda \
  --report .artifacts/regression-base-bundle.json

uv run --locked python scripts/check_regression.py compare \
  --name large --reference-dir .artifacts/baseline \
  --checkpoint .artifacts/checkpoints/seislm-large.ckpt \
  --trust-legacy-checkpoint --device cuda \
  --report .artifacts/regression-large-legacy.json
```

Repeat with the other model/source combinations. CUDA must be accessible when
requested. The default device is CPU. Comparison uses the matching reference
device: GPU outputs are compared against the original GPU outputs, rather than
requiring identical CPU/GPU rounding.

The checker covers contextual encoder outputs, convolutional features,
projected features, quantized features, codebook perplexity, and the contrastive,
diversity, and total masked losses. Loss evaluation fixes NumPy seeds to 42
separately for masks and negative indices, uses mask probability 0.2, mask length
10, minimum two masks, and 10 negatives per position. It also checks the public
`encode()` function and exact architecture configuration. Reports contain
absolute/relative numerical errors, exact-equality results, and provenance.
The default tolerances are `atol=1e-6`, `rtol=1e-5`; any failing check exits with
status 1. Investigate a mismatch before changing tolerances.

The initial portable-bundle checks passed with **exact tensor equality**, beyond
the configured tolerance, in all four model/device combinations:

| Released model | CPU maximum absolute error | GB10 CUDA maximum absolute error |
| --- | --- | --- |
| Base | 0 | 0 |
| Large | 0 | 0 |

This covers all eight reference fields and `encode()`; both architecture
configurations also matched exactly. The local reports are
`.artifacts/regression-base-bundle.json` and
`.artifacts/regression-large-bundle.json`. These results compare original
Lightning-loaded checkpoints against the exported bundles in PyTorch
`2.14.1+cu130`, CUDA 13.0, Python 3.12.3. They do not certify other hardware or
dependency combinations.

## Recorded results

Matching-environment comparisons of both pretrained releases against the original
loader passed with exact equality, including masked losses and `encode()`.
See [maintainer validation](validation.md#recorded-results) for the tested
versions, current package/notebook checks, and their limits. The original
checkpoints and generated reference tensors are not distributed in Git.

## Capturing a new independent baseline

The `capture` command deliberately uses the original Lightning loading route.
Run it with an environment that supports the original source and explicitly
select that checkout. Lightning 2.6 or newer is required for the explicit
legacy checkpoint deserialization option. Supply a saved, already normalized input tensor; its
values are shared by every model in the same reference directory.

```bash
BASELINE_REPO=/absolute/path/to/unmodified/seisLM
BASELINE_PYTHON=/absolute/path/to/baseline-environment/bin/python

"$BASELINE_PYTHON" scripts/check_regression.py capture \
  --source-root "$BASELINE_REPO" \
  --name base --checkpoint /absolute/path/to/seislm-base.ckpt \
  --trust-legacy-checkpoint \
  --input /absolute/path/to/normalized-waveforms.pt \
  --reference-dir /absolute/path/to/new-reference \
  --device cuda
```

The command records the loaded module path, Git commit when available,
checkpoint/input SHA-256 checksums, model configuration, parameter count, and
runtime versions. It refuses to overwrite a model's existing reference. Save a
complete dependency manifest alongside these records, for example with
`uv pip freeze --python "$BASELINE_PYTHON"`.

Only trust legacy `.ckpt` files whose source you know: the original Lightning
format contains pickled Python objects. The portable safetensors bundles do not
need that opt-in. The checker never downloads or publishes checkpoints.

## Relationship to the paper

[SeisLM, arXiv:2410.15765v1](https://arxiv.org/pdf/2410.15765v1) reports phase
benchmark curves on ETHZ, GEOFON, and STEAD at training fractions 5%, 10%, 20%,
50%, and 100%. Detection and phase identification use ROC AUC; P/S onset
regression uses RMSE in seconds. Figure 6 reports nine-class NRCA test accuracy:
58.33% for ConvNet, 65.11% for SeisLM-base, and 74.22% for SeisLM-large. The phase
curves do not provide an exact numeric results table suitable for strict checks.

The public Drive release contains pretrained base and large encoders. It does
not contain the trained phase-picking or NRCA heads needed to evaluate those
paper results directly. Historical plotting notebooks also reference a checkpoint
registry absent from the repository. Reproducing the downstream numbers requires
the corresponding task checkpoints, exact dataset versions and evaluation
targets, and split/preprocessing configuration. Retraining adds optimization
variability and is a separate experiment.

This refactor's release gate is therefore unchanged pretrained model behavior
plus installation/loading and execution of the two recommended demos. The
foreshock/aftershock tutorial is under review and excluded from this update. Its numerical comparisons do not establish
reproduction of the paper's downstream scores, fine-tuning quality, or performance
on continuous regional data.
