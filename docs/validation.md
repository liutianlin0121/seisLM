# Maintainer validation

The maintained demos are [pretrained inference and feature visualization](notebooks.md). Validate installation, numerical behavior, notebook execution, and browser interaction separately. These checks do not reproduce the paper's downstream scores or validate the paused foreshock/aftershock example.

## Repeatable checks

Run the package and inference suite from the locked environment:

```bash
uv run --locked --group dev --group notebook pytest
```

Python 3.11–3.12 includes the Transformers 4.44.2 reference tests. Python 3.13 skips the two cases requiring that older dependency; the remaining inference and runner tests still run.

For notebook changes, run base and large on a GPU, using separate output directories. From a clone:

```bash
uv run --locked --group notebook python scripts/run_notebook.py \
  --model base --device cuda --output-dir .artifacts/notebooks/base-cuda
uv run --locked --extra notebooks --group notebook python scripts/run_notebook.py \
  --notebook examples/demo_seislm_visualized_pretrained_features.ipynb \
  --model base --device cuda --output-dir .artifacts/notebooks/features-base-cuda
```

Repeat with `--model large` and new output paths. Use a GPU-enabled interpreter prepared according to the [installation guide](installation.md#environment); the lock uses CPU wheels on Linux x86-64. The runner uses its own interpreter for a fresh kernel, clears `PYTHONPATH`, and executes in a separate working directory. Each run retains the executed notebook, source hash, settings, package versions, timing, and error details in `execution-report.json`, plus `results/run-summary.json`. A cell error fails the command. The default per-cell timeout is 600 seconds; raise `--timeout` if a first download requires longer.

Keep notebook source outputs cleared. Model or preprocessing changes also require the [original-checkpoint regression comparison](regression.md), comparing each tested device against its matching reference. Retain the independent reference; do not recapture it from changed code to make a comparison pass.

### Public installation cell

Ordinary runs skip cells tagged `installation` and record `bootstrap_skipped`. For at least one release check, execute the complete installation cell in a disposable environment, outside the source checkout:

```bash
uv venv --seed --python 3.13 /tmp/seislm-notebook-check
uv pip install --python /tmp/seislm-notebook-check/bin/python nbclient nbformat ipykernel
```

Install a suitable GPU PyTorch build into that environment using the [official selector](https://pytorch.org/get-started/locally/), then run:

```bash
/tmp/seislm-notebook-check/bin/python scripts/run_notebook.py \
  --install --model base --device cuda \
  --output-dir .artifacts/notebooks/base-install
```

Repeat for feature visualization with `--notebook examples/demo_seislm_visualized_pretrained_features.ipynb`. `--install` changes the disposable environment: the notebook installs its pinned public wheel and plotting/data dependencies, and raises on installation failure. Verify that SeisLM imports from the installed wheel, rather than the checkout. A local installation check does not certify hosted Colab.

### Interactive browser behavior

After executing feature visualization, install Chromium once and check the exported files:

```bash
uv run --locked --extra notebooks --group notebook --group browser playwright install chromium
uv run --locked --extra notebooks --group notebook --group browser python scripts/check_interactive_visualization.py \
  --html .artifacts/notebooks/features-base-cuda/results/interactive_embeddings.html \
  --representations .artifacts/notebooks/features-base-cuda/results/representations.npz \
  --provenance .artifacts/notebooks/features-base-cuda/results/data-provenance.json \
  --check-duplicate-fragments \
  --output-dir .artifacts/notebooks/features-base-cuda/browser
```

The checker uses real mouse hovers in both projections and compares waveform values, sample times, selected metadata, linked highlights, and visible amplitudes against the saved data. It exercises offline standalone HTML, a sandboxed iframe, and two copies in one page. It retains screenshots and `browser-verification.json`; failures return a nonzero exit code. Repeat for the large-model export. This validates local browser behavior, not Colab's output renderer.

## Hosted Colab check

1. Open the current [inference](https://colab.research.google.com/drive/18dqsNcNsF9Kc9D53YPIVJMEjW_JX1ox7) or [feature-visualization](https://colab.research.google.com/drive/150HS1pyoY4FLnVRksc9E9Bw4UI2q7NtQ) demo. When testing changed source cells, upload the local notebook; an existing Drive copy does not update automatically.
2. Start a fresh Python 3 GPU runtime. Leave the model as base and set `DEVICE = "cuda"`. Run all cells, including installation, before importing scientific packages. If asked to restart, restart once and rerun from the top.
3. Confirm CUDA and the intended model revision. Inference should produce finite `[1, 749, 240]` features and aligned waveform/feature panels. In feature visualization, hover in both projections and confirm waveform and linked-highlight updates; download the HTML and check it after disconnecting the runtime.
4. Download the executed notebook and saved results, including `run-summary.json`. Record the date, runtime version, Python/PyTorch versions, GPU type, model revision, and any installation warnings. Distinguish a user-reported successful run from a retained execution report.
5. Repeat with large and a separate output directory. Inference should produce `[1, 749, 768]`; feature visualization should use the same windows.

## Recorded results

The following local checks were completed on 15 October 2026. Hardware was NVIDIA GB10 on Linux ARM64; it differs from hosted Colab.

| Check | Evidence and limits |
|---|---|
| Pre-push cleanup (`0.2.2.dev0`) | All 24 maintained interface tests passed on Python 3.12.3, including bundle paths containing `~`. Wheel and source distribution builds passed. The installed wheel, tested outside the checkout with base dependencies only, matched all 18 base/large CUDA reference tensors exactly, with maximum absolute error zero. The model configuration and all packaged source files matched. |
| Original checkpoint equivalence | All 36 reference tensor comparisons across base/large and CPU/CUDA were bitwise identical with matching PyTorch 2.14.1 environments. Architecture configurations also matched. The reference was captured from unmodified commit `ae1d27ac498ec7ae106e56eeaac386a0a5dd124f`; details and commands are in [regression validation](regression.md). |
| Earlier full-suite coverage | Before the cleanup, 30 tests passed on Python 3.12; 28 passed and two legacy Transformers cases skipped on Python 3.13. Installed-wheel and locked-environment runs were checked. |
| Inference notebooks | Base and large completed CPU/CUDA execution and public-wheel installation checks on Python 3.12/3.13. Outputs were finite with expected shapes. The accepted two-panel inference notebook also passed base GPU execution with Python 3.12.3, PyTorch 2.14.1+cu130, and NumPy 1.26.4. The author reported successful hosted Colab inference on Python 3.13.16; no automated hosted record was collected. |
| Interactive feature notebook | Base and large passed GPU execution using the public 0.2.1 wheel, Python 3.13.14, PyTorch 2.11.0+cu130, NumPy 2.1.3, and Plotly 7.1.0. The base run included the installation cell. All 19 previous base arrays and 31 large arrays, including t-SNE coordinates, were bitwise unchanged after adding interaction; data provenance was unchanged. |
| Interactive browser checks | Both model exports passed offline standalone, sandboxed iframe, and duplicate-output checks, with 16 real-pointer selections per model, no browser errors or external requests, and independent duplicate outputs. Hosted Colab hover rendering still requires a manual check. |

The feature demo's first uncached data check transferred 8.72 MiB of waveform data through 218 bounded range requests, in addition to the metadata/window tables. Its default 64 windows represented 30 distinct events. These are observations for the pinned data selection, not a general download-size guarantee.

Numerical equivalence is evaluated within matching environments. PyTorch 2.11 versus 2.14 produced small differences in the inference example: maximum absolute errors were `2.26e-6`/`7.18e-4` for base CPU/CUDA and `7.15e-6`/`1.44e-3` for large CPU/CUDA. Those arrays agreed at `atol=rtol=1e-3`; this observation does not replace the tighter original-reference comparison or define a tolerance for arbitrary hardware.

Local execution records are retained under ignored `.artifacts/notebooks/`, `.artifacts/colab313/`, `.artifacts/three-notebooks/`, and `.artifacts/interactive-notebook/`; the commands above reproduce new reports. Keep release records separately when publishing an update. Hosted GitHub Actions have not run for the unpushed changes. The public pretrained encoders have no released task heads with which to reproduce the paper's phase-picking or NRCA scores directly; see the [benchmark limitations](regression.md#relationship-to-the-paper).
