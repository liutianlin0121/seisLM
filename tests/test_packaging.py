"""Checks for the supported installation and data-location interfaces."""

import json
import os
from pathlib import Path
import subprocess
import sys

from importlib.resources import files


def test_configs_available_as_package_resources():
  """A wheel must carry configurations used by documented research scripts."""
  config = files("seisLM.configs").joinpath(
    "pretrain",
    "pretrain_config_std_norm_single_ax_8_datasets_32bit_scaleup_samp_false.json",
  )
  assert json.loads(config.read_text())["model_config"]["input_dim"] == 3


def test_paths_import_from_unrelated_working_directory(tmp_path):
  import seisLM

  # Source tests and wheel tests both execute the same subprocess assertion.
  # The source path is explicit so this also works without an editable install.
  package_parent = str(Path(seisLM.__file__).resolve().parent.parent)
  environment = os.environ.copy()
  environment.pop("SEISLM_HOME", None)
  environment["PYTHONPATH"] = package_parent
  result = subprocess.run(
    [sys.executable, "-c", (
      "from pathlib import Path; "
      "from seisLM.utils.project_path import DATA_DIR; "
      "assert Path(DATA_DIR).is_absolute()"
    )],
    cwd=tmp_path,
    env=environment,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
  assert list(tmp_path.iterdir()) == []


def test_explicit_data_root(tmp_path):
  environment = os.environ.copy()
  environment["SEISLM_HOME"] = str(tmp_path / "chosen-root")
  result = subprocess.run(
    [sys.executable, "-c", (
      "import json; from seisLM.utils.project_path import "
      "DATA_DIR, MODEL_SAVE_DIR; "
      "print(json.dumps([DATA_DIR, MODEL_SAVE_DIR]))"
    )],
    env=environment,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
  data_dir, models_dir = json.loads(result.stdout)
  assert Path(data_dir) == tmp_path / "chosen-root" / "data"
  assert Path(models_dir) == tmp_path / "chosen-root" / "results" / "models"
  assert not (tmp_path / "chosen-root").exists()
