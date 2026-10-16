# Practitioner notebooks

Start with inference, then explore the pretrained representations. Both introductory notebooks run in Colab or locally, install a released wheel, and load pinned public Hugging Face models. No Hugging Face token, Drive mount, W&B account, original checkpoint, or repository clone is needed in Colab. For fine-tuning, see the [foreshock/aftershock experiment](#original-foreshockaftershock-experiment).

| Demo | Open in Colab | Notebook source |
|---|---|---|
| Pretrained inference | [Run inference](https://colab.research.google.com/drive/18dqsNcNsF9Kc9D53YPIVJMEjW_JX1ox7) | [Inference notebook](../examples/demo_pretrained_inference.ipynb) |
| Visualize pretrained features | [Explore features](https://colab.research.google.com/drive/150HS1pyoY4FLnVRksc9E9Bw4UI2q7NtQ) | [Feature notebook](../examples/demo_seislm_visualized_pretrained_features.ipynb) |

## Run in Colab

Select **Python 3**, choose a GPU under **Runtime > Change runtime type**, and **Run all**. Python 3.11–3.13 is supported; there is no need to choose a particular minor version. Restart once only if installation requests it, then run from the top again.

Edit the settings cell to choose `MODEL_NAME` (`"base"` or `"large"`), `DEVICE`, and `OUTPUT_DIR`. `DEVICE = "auto"` uses CUDA when available and otherwise uses the CPU. Download saved results through Colab's **Files** pane. Models and data are cached within a runtime; a fresh runtime may download them again.

## Pretrained inference

The notebook uses ObsPy's bundled BW.RJOB recording: three aligned Z/N/E components sampled at 100 Hz, with 3,000 samples each. It normalizes each component, extracts contextual features, and plots the input waveform above the features with a shared time axis. Expected feature shapes are `[1, 749, 240]` for base and `[1, 749, 768]` for large.

Saved results include `contextual_features.npy`, `token_times.npy`, `waveforms.png`, `features.png`, and `run-summary.json`.

To use your own data, replace the loading cell and follow the [input contract](installation.md#input-and-output-contract), including sampling rate, component orientation, alignment, and gap handling. Features are learned representations, not calibrated P/S probabilities or event detections.

## Feature visualization

The notebook compares a pretrained encoder with the same architecture initialized at random, using real ETHZ earthquake/noise windows. By default it selects 32 complete development records, pairing one earthquake and one noise window per record for 64 windows. It downloads about 17 MB of metadata/window definitions, then only the required waveform bytes through HTTP range requests, rather than the full approximately 23 GB waveform file. Data are cached in `~/.cache/seislm/ethz-demo`.

Complete 100 or 200 Hz ZNE records are selected; 200 Hz windows are filtered and downsampled to 100 Hz. Both encoders receive the same normalized 3,000-sample windows. ETHZ was part of pretraining, but its development split was excluded.

The notebook pools features from every layer and projects the selected layer with t-SNE. The two projections are fitted separately: positions, axes, and distances cannot be compared across panels. Change `SAMPLES_PER_CLASS` (8–64, default 32) or `HIDDEN_LAYER` (default `-1`, the last layer) in the settings cell to explore another selection.

Hover over a point in either projection to inspect its normalized Z/N/E waveform and highlight the same window in both panels. Download `interactive_embeddings.html` to explore it offline, even after the notebook kernel stops. The HTML embeds Plotly and the waveform data.

Other saved results are `representations.npz` (all pooled layers, t-SNE coordinates, labels, and normalized input waveforms), `data-provenance.json`, `example_windows.png`, `embedding_comparison.png`, and `run-summary.json`.

## Run locally

From a clone, execute inference using the locked notebook dependencies:

```bash
uv run --locked --group notebook python scripts/run_notebook.py \
  --model base --device cuda --output-dir .artifacts/notebooks/base-cuda
```

Feature visualization needs the additional `notebooks` extra:

```bash
uv run --locked --extra notebooks --group notebook python scripts/run_notebook.py \
  --notebook examples/demo_seislm_visualized_pretrained_features.ipynb \
  --model base --device cuda --output-dir .artifacts/notebooks/features-base-cuda
```

Use `--model large` for the larger encoder or `--device cpu` without a GPU. An explicit `--device cuda` fails if CUDA is unavailable. The lock selects a CPU PyTorch build on Linux x86-64; see the [installation guide](installation.md#environment) to prepare a GPU environment there.

The runner saves `executed.ipynb`, `execution-report.json`, and `results/` in the output directory and returns a nonzero exit code on a cell error. Installation cells are skipped by default because the locked dependencies are already installed. The [maintainer validation guide](validation.md) explains installation-cell checks, browser checks, and hosted Colab verification.

## Model versions

The notebooks use immutable revisions:

| Artifact | Revision |
|---|---|
| Base encoder | `a9e9f79af940a3ea28a49ef9fa495ae89b48d576` |
| Large encoder | `3ad2a723faeded0ca45b555c9c0ea8823affcc4f` |
| SeisLM 0.2.1 wheel, hosted with base | `13a3dcd6ae56c39f5b0750b817c8788a7fc3c99b` |

The wheel pin is independent of the selected encoder. See [Hugging Face releases](huggingface.md) for provenance and [validation](validation.md#recorded-results) for the verification status.

## Original foreshock/aftershock experiment

Run the [classification demo in Colab](https://colab.research.google.com/drive/1QYxAyBi3h8P0m_FZ4ajeJ5oel5X1mrii).

The [classification notebook](../examples/demo_seislm_foreshock_aftershock_classification.ipynb) follows the original SeisLM-base experiment: all 5,580 training waveforms, 15 epochs, batch size 32, and the original head, augmentation, optimizer, and scheduler. It requires a GPU. As in the original demo, all 1,593 `test` waveforms supply training monitoring and final evaluation; the 792-waveform `val` split is unused. Final testing and the percentage confusion matrix use `last.ckpt`.

```bash
uv run --locked --extra notebooks --group notebook python scripts/run_notebook.py \
  --notebook examples/demo_seislm_foreshock_aftershock_classification.ipynb \
  --model base --device cuda --timeout 7200 \
  --output-dir .artifacts/notebooks/foreshock-original-cuda
```

The longer timeout allows the full training cell to finish. See [experiment validation](validation.md#foreshockaftershock-experiment) for comparison with the original recipe and recorded results. Hosted Colab verification of this restored notebook is pending.
