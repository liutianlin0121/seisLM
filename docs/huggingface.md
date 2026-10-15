# Hugging Face releases and checkpoint export

The existing pretrained models are published as portable safetensors bundles under the MIT license:

| Model | Repository | Release revision |
|---|---|---|
| Base | [ltlltl2006/seislm-base](https://huggingface.co/ltlltl2006/seislm-base/tree/a9e9f79af940a3ea28a49ef9fa495ae89b48d576) | `a9e9f79af940a3ea28a49ef9fa495ae89b48d576` |
| Large | [ltlltl2006/seislm-large](https://huggingface.co/ltlltl2006/seislm-large/tree/3ad2a723faeded0ca45b555c9c0ea8823affcc4f) | `3ad2a723faeded0ca45b555c9c0ea8823affcc4f` |

Use these revisions to reproduce the examples. Public downloads do not require a Hugging Face account:

```python
from seisLM import load_pretrained

model = load_pretrained("base", revision="a9e9f79af940a3ea28a49ef9fa495ae89b48d576")
```

The model-weight revisions above remain pinned for reproducibility. A later compatibility release adds `seislm-0.2.1-py3-none-any.whl` for Python 3.11–3.13, including Colab's Python 3.13 / NumPy 2 environment:

| Repository | Package 0.2.1 release revision |
|---|---|
| Base | `13a3dcd6ae56c39f5b0750b817c8788a7fc3c99b` |
| Large | `0ac6158dbfd139591f57d3064a893e667387106f` |

The model weights, configuration, preprocessing, and original wheel are unchanged in these commits. The [installation guide](installation.md#uv-and-pip) and [inference notebook](../examples/demo_pretrained_inference.ipynb) use the wheel from the pinned base compatibility revision and the original verified weight revisions. This is ordinary Python packaging; it does not require Transformers or a Hub-specific model implementation.

## Prepare local artifacts

For a future package release, choose a new, unpublished version in `pyproject.toml` and regenerate `uv.lock` before building. Keep previously published wheels, including `seislm-0.2.1-py3-none-any.whl`, unchanged; never upload different bytes under an existing wheel filename.

Download the original checkpoints as described in [installation](installation.md), then run from the repository root:

```bash
uv build --wheel --out-dir .artifacts/hub-next/wheels
seislm_release_version=$(uv run --locked python -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])')
seislm_release_wheel=".artifacts/hub-next/wheels/seislm-${seislm_release_version}-py3-none-any.whl"
uv run --locked python scripts/prepare_hub.py --checkpoint .artifacts/checkpoints/seislm-base.ckpt --output .artifacts/hub-next/seislm-base --model base --trust-legacy-checkpoint --wheel "$seislm_release_wheel"
uv run --locked python scripts/prepare_hub.py --checkpoint .artifacts/checkpoints/seislm-large.ckpt --output .artifacts/hub-next/seislm-large --model large --trust-legacy-checkpoint --wheel "$seislm_release_wheel"
```

Use empty model output directories; the exporter refuses to mix files with a previous release. The commands select the wheel matching the source package version. The script verifies the original checkpoint's official SHA-256 before reading its legacy Python serialization. The export contains:

- `model.safetensors`: all existing pretraining model weights, with no optimizer state.
- `config.json`: architecture configuration, bundle format, and source checkpoint provenance.
- `preprocessing.json`: the waveform input specification.
- `README.md`: a model card explaining usage and limitations.
- `manifest.json`: checksums of the exported weights and configuration.

The released folders also include the MIT license and the inference package wheel. Exporting does not fine-tune or change the model. Keep the original Lightning checkpoint for original training resumption. These portable artifacts are pretrained encoders with pretraining heads, not fine-tuned phase pickers.

Confirm local loading before publishing:

```bash
uv run --locked python examples/quickstart.py --checkpoint .artifacts/hub-next/seislm-base --device auto
uv run --locked python examples/quickstart.py --checkpoint .artifacts/hub-next/seislm-large --device auto
```

See [regression validation](regression.md) for comparisons between original checkpoints and exports. Successful export alone does not reproduce the task benchmarks in the paper.

## Maintainer authentication

Use your own terminal for authentication. Do not put an access token in chat, a notebook, a command-line argument, or a committed file.

```bash
uv run --locked hf auth login
uv run --locked hf auth whoami
```

Sign in as `ltlltl2006`, using a token with permission to create and write the intended model repositories. The [Hugging Face token settings](https://huggingface.co/settings/tokens) and [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) explain account authentication. Standard Hub caching and authentication stay outside the model bundle.

## Publishing a future revision

The author has confirmed that the pretrained weights use the MIT license. For future revisions, review each generated model card for the correct license, training-data description, citation, preprocessing, and limitations before publication.

The repositories above already exist. Upload reviewed folders from your terminal when publishing an intentional update:

```bash
uv run --locked hf upload ltlltl2006/seislm-base .artifacts/hub-next/seislm-base .
uv run --locked hf upload ltlltl2006/seislm-large .artifacts/hub-next/seislm-large .
```

These commands publish a new commit to the selected repositories. Preparing local artifacts alone does not perform an upload. Existing revision-pinned examples continue to use their original snapshots.

## Verify the published artifacts

Record each upload's commit SHA. Test a clean download pinned to that commit, rather than relying only on a local export:

```bash
uv run --locked python examples/quickstart.py --checkpoint ltlltl2006/seislm-base --revision a9e9f79af940a3ea28a49ef9fa495ae89b48d576
uv run --locked python examples/quickstart.py --checkpoint ltlltl2006/seislm-large --revision 3ad2a723faeded0ca45b555c9c0ea8823affcc4f
```

The downloader resolves weights and configuration from the same snapshot and caches the download. A specific commit keeps examples reproducible if a repository changes later. Follow with `--local-files-only` to check cached offline loading.

The commands above verify this release; substitute each new upload's SHA when checking a future revision. After those checks, update the model catalog links and notebook downloads to the verified revisions. Keep the package wheel compatible with the model bundle and preserve the original Drive links for continuity. Private or gated downloads require the corresponding access and authentication.

The Hub's [download guide](https://huggingface.co/docs/huggingface_hub/guides/download) and [model-card guide](https://huggingface.co/docs/hub/model-cards) describe revision pinning and model documentation. A Transformers conversion is not required to host these custom PyTorch models.
