#!/usr/bin/env python3
"""Execute a demo in an isolated working directory and retain a run report.

Example:
  uv run --locked --group notebook python scripts/run_notebook.py \
    --device cpu --model base --output-dir .artifacts/notebooks/base-cpu

The current Python interpreter runs the notebook kernel. Installation cells are
skipped unless --install is supplied; this permits testing either an existing
locked environment or the notebook's complete installation instructions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback


DEFAULT_NOTEBOOK = (
  Path(__file__).resolve().parents[1]
  / "examples"
  / "demo_pretrained_inference.ipynb"
)


def positive_seconds(value):
  """Reject zero/negative per-cell timeouts instead of silently disabling them."""
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("timeout must be a positive number of seconds")
  return number


def make_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
  parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
  parser.add_argument("--model", choices=("base", "large"), default="base")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--install",
    action="store_true",
    help="also execute installation cells; use a disposable environment with pip",
  )
  parser.add_argument(
    "--timeout", type=positive_seconds, default=600,
    help="maximum execution time per cell in seconds (default: 600)",
  )
  return parser


def _check_installation_environment():
  if importlib.util.find_spec("pip") is None:
    raise ValueError(
      "--install needs pip in the current interpreter. Create a disposable "
      "environment with `uv venv --seed PATH`, install nbclient, nbformat, and "
      "ipykernel there, then run this command with PATH/bin/python. "
      "Omit --install to test the locked environment without reinstalling it."
    )


def run_notebook(args):
  """Return a JSON-compatible report; failed cells are retained, never ignored."""
  try:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
  except ImportError as error:
    raise ValueError(
      "Notebook execution dependencies are missing. Run this command with "
      "`uv run --locked --group notebook python scripts/run_notebook.py ...`."
    ) from error

  notebook_path = args.notebook.resolve(strict=True)
  if not notebook_path.is_file() or notebook_path.suffix != ".ipynb":
    raise ValueError("--notebook must name an existing .ipynb file")
  output_dir = args.output_dir.resolve()
  if notebook_path == output_dir / "executed.ipynb":
    raise ValueError("Choose an output directory that does not overwrite the source notebook")
  if args.install:
    _check_installation_environment()

  source = notebook_path.read_bytes()
  notebook = nbformat.reads(source.decode("utf-8"), as_version=4)
  installation_cells = [
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code"
    and "installation" in cell.metadata.get("tags", [])
  ]
  for cell in notebook.cells:
    if cell.cell_type == "code":
      cell.outputs = []
      cell.execution_count = None

  work_dir = output_dir / "work"
  results_dir = output_dir / "results"
  work_dir.mkdir(parents=True, exist_ok=True)
  results_dir.mkdir(parents=True, exist_ok=True)
  summary_path = results_dir / "run-summary.json"
  # Keep an earlier summary out of a failed rerun's report.
  old_summary_stat = summary_path.stat() if summary_path.exists() else None

  environment = os.environ.copy()
  environment.pop("PYTHONPATH", None)
  environment.update({
    "SEISLM_NOTEBOOK_DEVICE": args.device,
    "SEISLM_NOTEBOOK_MODEL": args.model,
    "SEISLM_NOTEBOOK_OUTPUT_DIR": str(results_dir),
    "MPLBACKEND": "Agg",
  })
  started = datetime.now(timezone.utc)
  started_clock = time.monotonic()
  report = {
    "status": "running",
    "source_notebook": str(notebook_path),
    "source_sha256": hashlib.sha256(source).hexdigest(),
    "executed_notebook": str(output_dir / "executed.ipynb"),
    "working_directory": str(work_dir),
    "results_directory": str(results_dir),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "device_requested": args.device,
    "model_requested": args.model,
    "cell_timeout_seconds": args.timeout,
    "bootstrap_skipped": bool(installation_cells) and not args.install,
    "installation_cells_skipped": installation_cells if not args.install else [],
    "installation_requested": args.install,
    "started_at": started.isoformat(),
    "hosted_colab_verified": False,
    "notebook_summary": None,
    "error": None,
  }
  try:
    # A local, temporary kernelspec selects this interpreter without registering
    # a user/global kernel or trusting a notebook's arbitrary kernel name.
    with tempfile.TemporaryDirectory(prefix="seislm-notebook-kernel-") as kernel_dir:
      kernel_name = "seislm-notebook"
      spec_dir = Path(kernel_dir) / kernel_name
      spec_dir.mkdir()
      (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "SeisLM notebook execution",
        "language": "python",
      }))
      manager = KernelManager(
        kernel_name=kernel_name,
        kernel_spec_manager=KernelSpecManager(
          kernel_dirs=[kernel_dir], ensure_native_kernel=False,
        ),
      )
      client = NotebookClient(
        notebook,
        km=manager,
        timeout=args.timeout,
        allow_errors=False,
        force_raise_errors=True,
        skip_cells_with_tag="" if args.install else "installation",
        resources={"metadata": {"path": str(work_dir)}},
      )
      # Passing a manager makes nbclient consider it caller-owned. Explicitly
      # request cleanup on both successful execution and failed cells.
      client.execute(env=environment, cleanup_kc=True)
    report["status"] = "success"
  except Exception as error:  # Include cell traceback in report and notebook.
    report["status"] = "failed"
    report["error"] = {
      "type": type(error).__name__,
      "message": str(error),
      "traceback": traceback.format_exc(),
    }
  finally:
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started_clock, 3)
    report["executed_code_cells"] = sum(
      cell.cell_type == "code" and cell.execution_count is not None
      for cell in notebook.cells
    )
    report["cell_errors"] = [
      {"cell_index": index, "name": output.ename, "message": output.evalue}
      for index, cell in enumerate(notebook.cells)
      if cell.cell_type == "code"
      for output in cell.outputs
      if output.output_type == "error"
    ]
    nbformat.write(notebook, output_dir / "executed.ipynb")
    if summary_path.exists():
      new_summary_stat = summary_path.stat()
      if old_summary_stat is None or (
        new_summary_stat.st_mtime_ns != old_summary_stat.st_mtime_ns
        or new_summary_stat.st_size != old_summary_stat.st_size
      ):
        try:
          report["notebook_summary"] = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
          report["summary_read_error"] = str(error)
    (output_dir / "execution-report.json").write_text(
      json.dumps(report, indent=2) + "\n"
    )
  return report


def main(argv=None):
  parser = make_parser()
  args = parser.parse_args(argv)
  try:
    report = run_notebook(args)
  except (OSError, ValueError) as error:
    parser.exit(2, f"error: {error}\n")
  print(f"Notebook {report['status']}: {args.output_dir.resolve() / 'execution-report.json'}")
  if report["error"]:
    print(report["error"]["traceback"], file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
