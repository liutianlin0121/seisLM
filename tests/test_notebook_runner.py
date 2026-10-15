"""Exercise the runner's real kernel, isolation, and failure reporting."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_notebook.py"
spec = importlib.util.spec_from_file_location("notebook_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_notebook(path, cells):
  nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
  return path


def arguments(notebook_path, output_dir, *extra):
  return runner.make_parser().parse_args([
    "--notebook", str(notebook_path), "--output-dir", str(output_dir),
    "--device", "cpu", "--model", "large", *extra,
  ])


def test_kernel_uses_current_interpreter_and_retains_provenance(tmp_path, monkeypatch):
  """A passing run must not depend on cwd imports or execute bootstrap by default."""
  monkeypatch.setenv("PYTHONPATH", "/invalid/source/import/path")
  notebook_path = write_notebook(tmp_path / "source.ipynb", [
    nbformat.v4.new_code_cell(
      "raise RuntimeError('installation cell must be skipped')",
      metadata={"tags": ["installation"]},
    ),
    nbformat.v4.new_code_cell(
      "import json, os, pathlib, sys\n"
      "summary = {'python_executable': sys.executable,\n"
      "           'cwd': str(pathlib.Path.cwd()),\n"
      "           'pythonpath': os.environ.get('PYTHONPATH'),\n"
      "           'device': os.environ['SEISLM_NOTEBOOK_DEVICE'],\n"
      "           'model': os.environ['SEISLM_NOTEBOOK_MODEL']}\n"
      "output = pathlib.Path(os.environ['SEISLM_NOTEBOOK_OUTPUT_DIR'])\n"
      "(output / 'run-summary.json').write_text(json.dumps(summary))\n"
      "print('real kernel executed')\n"
    ),
  ])
  original = notebook_path.read_bytes()
  output_dir = tmp_path / "run"
  report = runner.run_notebook(arguments(notebook_path, output_dir))
  assert report["status"] == "success", report["error"]
  assert report["bootstrap_skipped"] is True
  assert report["installation_cells_skipped"] == [0]
  assert report["hosted_colab_verified"] is False
  assert report["notebook_summary"] == {
    "python_executable": sys.executable,
    "cwd": str(output_dir / "work"),
    "pythonpath": None,
    "device": "cpu",
    "model": "large",
  }
  executed = nbformat.read(output_dir / "executed.ipynb", as_version=4)
  assert executed.cells[0].execution_count is None
  assert executed.cells[1].outputs[0].text == "real kernel executed\n"
  assert notebook_path.read_bytes() == original
  assert json.loads((output_dir / "execution-report.json").read_text()) == report


@pytest.mark.parametrize("tags", [[], ["raises-exception"]], ids=["plain", "tagged"])
def test_cell_failure_is_nonzero_and_preserves_traceback(tmp_path, capsys, tags):
  notebook_path = write_notebook(tmp_path / "source.ipynb", [
    nbformat.v4.new_code_cell(
      "raise RuntimeError('meaningful failure')", metadata={"tags": tags},
    ),
    nbformat.v4.new_code_cell("raise AssertionError('must not execute')"),
  ])
  output_dir = tmp_path / "run"
  assert runner.main([
    "--notebook", str(notebook_path), "--output-dir", str(output_dir),
  ]) == 1
  report = json.loads((output_dir / "execution-report.json").read_text())
  assert report["status"] == "failed"
  assert report["cell_errors"] == [{
    "cell_index": 0, "name": "RuntimeError", "message": "meaningful failure",
  }]
  assert "meaningful failure" in report["error"]["traceback"]
  executed = nbformat.read(output_dir / "executed.ipynb", as_version=4)
  assert executed.cells[0].outputs[0].ename == "RuntimeError"
  assert executed.cells[1].execution_count is None
  assert "meaningful failure" in capsys.readouterr().err


def test_actual_notebook_bootstrap_stops_on_pip_failure(tmp_path, monkeypatch):
  """Exercise the maintained install cell without installing or downloading."""
  source = nbformat.read(runner.DEFAULT_NOTEBOOK, as_version=4)
  bootstrap = next(
    cell for cell in source.cells
    if cell.cell_type == "code"
    and "installation" in cell.metadata.get("tags", [])
  )
  # Reject shell magics before execution: the test only intercepts subprocess.run.
  compile(bootstrap.source, str(runner.DEFAULT_NOTEBOOK), "exec")
  notebook_path = write_notebook(tmp_path / "source.ipynb", [
    nbformat.v4.new_code_cell(
      "import subprocess, sys\n"
      "original_run = subprocess.run\n"
      "def fail_pip(command, *args, **kwargs):\n"
      "    if command[:4] == [sys.executable, '-m', 'pip', 'install']:\n"
      "        if kwargs.get('check', False):\n"
      "            raise subprocess.CalledProcessError(23, command)\n"
      "        return subprocess.CompletedProcess(command, returncode=23)\n"
      "    return original_run(command, *args, **kwargs)\n"
      "subprocess.run = fail_pip\n"
    ),
    bootstrap,
    nbformat.v4.new_code_cell("raise AssertionError('analysis must not execute')"),
  ])
  # A real pip module is unnecessary because its subprocess is intercepted.
  monkeypatch.setattr(runner, "_check_installation_environment", lambda: None)
  output_dir = tmp_path / "run"
  assert runner.main([
    "--notebook", str(notebook_path), "--output-dir", str(output_dir), "--install",
  ]) == 1
  report = json.loads((output_dir / "execution-report.json").read_text())
  assert report["status"] == "failed"
  assert report["bootstrap_skipped"] is False
  assert len(report["cell_errors"]) == 1
  assert report["cell_errors"][0]["cell_index"] == 1
  assert report["cell_errors"][0]["name"] == "CalledProcessError"
  assert "23" in report["cell_errors"][0]["message"]
  executed = nbformat.read(output_dir / "executed.ipynb", as_version=4)
  assert executed.cells[2].execution_count is None


def test_install_option_executes_bootstrap_and_checks_for_pip(tmp_path, monkeypatch):
  notebook_path = write_notebook(tmp_path / "source.ipynb", [
    nbformat.v4.new_code_cell(
      "raise RuntimeError('bootstrap was executed')",
      metadata={"tags": ["installation"]},
    ),
  ])
  args = arguments(notebook_path, tmp_path / "run", "--install")
  # No pip mutation is needed to prove that the unchanged bootstrap is executed.
  monkeypatch.setattr(runner, "_check_installation_environment", lambda: None)
  report = runner.run_notebook(args)
  assert report["status"] == "failed"
  assert report["bootstrap_skipped"] is False
  assert report["cell_errors"][0]["message"] == "bootstrap was executed"
  monkeypatch.undo()
  monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
  with pytest.raises(ValueError, match="uv venv --seed"):
    runner._check_installation_environment()


def test_arguments_reject_unsupported_device_and_disabled_timeout(tmp_path):
  parser = runner.make_parser()
  with pytest.raises(SystemExit):
    parser.parse_args(["--output-dir", str(tmp_path), "--device", "gpu"])
  with pytest.raises(SystemExit):
    parser.parse_args(["--output-dir", str(tmp_path), "--timeout", "0"])
