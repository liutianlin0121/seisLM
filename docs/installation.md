# Installation and pretrained inference

## Environment

Use Python 3.11–3.13 for pretrained inference. The repository's `pyproject.toml` describes the installable package; `uv.lock` records the maintainer environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from a clone:

```bash
uv sync --locked
uv run --locked python -c 'import seisLM; print(seisLM.__version__)'
```

The lock configuration selects a CUDA 13.0 PyTorch build on Linux ARM64 and a CPU build on Linux x86-64. A working GPU driver is also required. `uv run --locked` uses this environment consistently; it does not require manually activating a Conda environment.

For an NVIDIA GPU on Linux x86-64, select a compatible PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/) in a separate environment, then install SeisLM into that environment. The locked Linux x86-64 workflow uses CPU wheels. Changing a CUDA build is a deliberate environment change, rather than an extra `pip install` into the locked environment. Other platform combinations have not been validated by this refactor.

Additional dependencies are optional:

```bash
uv sync --locked --group dev
uv sync --locked --group notebook
uv sync --locked --extra notebooks --group notebook
uv sync --locked --extra training --extra notebooks --group dev
```

Pretrained inference does not require Lightning, SeisBench, W&B, torchtune, or the training dependencies for the released LayerNorm models. The `notebook` group adds execution, plotting, and ObsPy example-data dependencies. Add the `notebooks` extra for feature visualization; it includes the research-data and downstream-training dependencies. Neither maintained demo needs a W&B account, a repository clone in Colab, or a legacy checkpoint.

The `training` extra supplies dependencies used by downstream training code; it is not a complete environment for reproducing original pretraining. The original pretraining collator imports private Transformers APIs and needs a separate Python 3.11–3.12 research environment with Transformers 4.44.2, the original configurations, datasets, and checkpoints. Experimental RMSNorm configurations also need torchtune. Original pretraining has not been revalidated end to end in the maintained environment; keep its environment separate from pretrained inference.

## uv and pip

`uv` manages the maintained development environment and its lockfile. For inference without cloning the repository, create an environment in a new directory and install the versioned wheel hosted with the base model:

```bash
uv venv --python 3.13
uv pip install "https://huggingface.co/ltlltl2006/seislm-base/resolve/13a3dcd6ae56c39f5b0750b817c8788a7fc3c99b/seislm-0.2.1-py3-none-any.whl"
uv run --no-project python -c 'from seisLM import load_pretrained; model = load_pretrained("base", revision="a9e9f79af940a3ea28a49ef9fa495ae89b48d576"); print(model.config.hidden_size)'
```

For NVIDIA GPUs, first install the appropriate PyTorch build in this environment using the [official selector](https://pytorch.org/get-started/locally/); the wheel does not choose a CUDA variant for you. This standalone installation uses compatible dependency ranges rather than the repository's exact lockfile. The notebook uses the PyTorch already supplied by Colab when it satisfies those ranges.

Packaging remains standard Python packaging, so the same wheel can also be consumed by pip in an existing environment. For building from a checkout:

```bash
uv build
# In the downstream user's environment:
python -m pip install dist/seislm-*.whl
```

This does not introduce a separate pip requirements list. A pip installation resolves the compatible dependency ranges in package metadata; `uv sync --locked` uses the exact lockfile resolutions. PyPI installation should only be documented against a package version after that version has actually been published.

SeisLM 0.2.1 supports Python 3.11–3.13. Python 3.11–3.12 retain NumPy 1.x; Python 3.13 uses NumPy 2.1 or newer. The wheel selects these constraints automatically. The optional training dependencies select SeisBench 0.8 on Python 3.11–3.12 and 0.9 on Python 3.13 for NumPy compatibility. The maintained notebook checks are recorded separately from the original research experiments, which have not all been revalidated on Python 3.13.

The two [practitioner notebooks](notebooks.md) install the public 0.2.1 wheel and load pinned Hugging Face model revisions. The source checkout is versioned separately as unreleased `0.2.2.dev0`; it does not replace that immutable wheel. Start with [inference in Colab](https://colab.research.google.com/drive/18dqsNcNsF9Kc9D53YPIVJMEjW_JX1ox7), then try [feature visualization in Colab](https://colab.research.google.com/drive/150HS1pyoY4FLnVRksc9E9Bw4UI2q7NtQ). Their installation cells handle notebook-specific dependencies. Local execution, installation checks, browser checks, and the hosted Colab checklist are documented in [maintainer validation](validation.md). The original foreshock/aftershock notebook remains research code while its modernization and accuracy review are paused.

## Obtain a checkpoint

The recommended inference artifacts are the portable [base](https://huggingface.co/ltlltl2006/seislm-base/tree/a9e9f79af940a3ea28a49ef9fa495ae89b48d576) and [large](https://huggingface.co/ltlltl2006/seislm-large/tree/3ad2a723faeded0ca45b555c9c0ea8823affcc4f) Hub bundles. The loading API downloads and caches their safetensors weights and JSON configuration together. Public downloads do not require authentication.

```python
from seisLM import load_pretrained

model = load_pretrained("base", revision="a9e9f79af940a3ea28a49ef9fa495ae89b48d576")
```

The original [Google Drive checkpoint folder](https://drive.google.com/drive/folders/194vColWfn0utQbHP04PAt2efKO40Z3Os) remains available for compatibility with the research code. The following commands download the official base or large pretrained checkpoints:

```bash
mkdir -p .artifacts/checkpoints
uv run --locked --with gdown gdown 12L2Js1DIXCIxacQJ4K0hRKJ6qHFrQ7K- -O .artifacts/checkpoints/seislm-base.ckpt
uv run --locked --with gdown gdown 1LPUCltHsle4OqDPGEtcHPdoRnLppY-KH -O .artifacts/checkpoints/seislm-large.ckpt
```

| Checkpoint | SHA-256 |
|---|---|
| Base | `6e97c53c562e9ba6bc96df85d8d208b89ff84e80a25468825f29bb5bace5eacb` |
| Large | `db44b1a46512de2cdf5c310c7ccfdac0ef04931cc89b7768fad41c978266df4b` |

Legacy Lightning checkpoints contain pickled Python objects. The quickstart verifies an official checksum before loading them; the general API requires explicit `trust_legacy_checkpoint=True`. Only use that option for a checkpoint whose source you trust. [Exporting a portable bundle](huggingface.md) preserves the weights in safetensors and the configuration in JSON, and avoids that legacy deserialization on subsequent loads.

## Run the quickstart

```bash
uv run --locked python examples/quickstart.py --checkpoint base --revision a9e9f79af940a3ea28a49ef9fa495ae89b48d576
uv run --locked python examples/quickstart.py --checkpoint .artifacts/checkpoints/seislm-base.ckpt --trust-legacy-checkpoint
uv run --locked python examples/quickstart.py --checkpoint .artifacts/hub-next/seislm-base --device cpu
```

The first command downloads the portable release; the second uses a previously downloaded original checkpoint; the third requires that you have already exported the local portable bundle. `--device auto` is the default and selects CUDA if available. To supply your own waveform, add `--waveform path/to/waveform.npy`. The array must be float32 with shape `[3, samples]` or `[batch, 3, samples]`, sampled at 100 Hz with ZNE component order. The script demeans and normalizes each component before extracting features. It does not read MiniSEED, rotate components, resample, filter, or fill gaps.

## Input and output contract

| Property | Released pretrained models |
|---|---|
| Tensor shape | `[batch, 3, samples]` |
| Tensor type | Finite `torch.float32` values |
| Component order | Vertical, North, East (`ZNE`) |
| Sampling rate | 100 Hz |
| Reference example length | 3001 samples (other lengths must be long enough for the convolutional encoder) |
| Normalization | Demean each component over time, divide by its population standard deviation plus `1e-10`; replace a zero standard deviation with 1 before division |
| Feature output | `[batch, reduced_time, hidden_size]`; hidden size is 240 for base and 768 for large |

`load_pretrained()` returns an evaluation-mode pretraining model. `encode()` extracts contextual features without gradients, validates shape/type/finite values, and moves the tensor to the model's device. Neither function infers sampling rate or component orientation from tensor values. They do not normalize, resample, filter, remove instrument response, or fill missing components.

Supply complete three-component records for deterministic inference. The original training pipeline randomly copies a present component into a zero-variance channel; that augmentation is not silently applied by the new inference API. No automatic physical-unit conversion or inference of instrument units is performed. Contextual features are not calibrated P/S probabilities, event decisions, or absolute timestamps.

## Loading and offline use

```python
from seisLM import load_pretrained

model = load_pretrained("/absolute/path/to/exported-bundle", device="cpu")
```

A portable bundle contains `config.json`, `model.safetensors`, and `preprocessing.json`. Keep these files together. Local bundles do not require Hugging Face authentication or network access.

A hosted load can be pinned to its commit:

```python
model = load_pretrained(
    "ltlltl2006/seislm-base",
    revision="a9e9f79af940a3ea28a49ef9fa495ae89b48d576",
    cache_dir="/path/to/model-cache",
    device="cuda",
)
# Subsequent use of the cached snapshot, with downloads disabled:
model = load_pretrained(
    "ltlltl2006/seislm-base",
    revision="a9e9f79af940a3ea28a49ef9fa495ae89b48d576",
    cache_dir="/path/to/model-cache",
    local_files_only=True,
)
```

Catalog aliases `"base"` and `"large"` resolve to `ltlltl2006/seislm-base` and `ltlltl2006/seislm-large`. The large model's release revision is `3ad2a723faeded0ca45b555c9c0ea8823affcc4f`.
