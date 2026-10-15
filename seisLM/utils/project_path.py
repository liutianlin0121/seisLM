"""Locations for research datasets and results.

Set ``SEISLM_HOME`` to place these directories somewhere explicit. Source
checkouts retain their historical repository-local defaults; installed wheels
use the current working directory. Importing this module creates no files and
does not require Git or a Git checkout.
"""

import glob
import os
from pathlib import Path
from typing import Optional, List


def gitdir() -> str:
  """Return the working root (historical name retained for compatibility).

  Unlike the original helper, this works in an installed package and does not
  accidentally select an unrelated Git repository containing the working dir.
  """
  configured_root = os.environ.get("SEISLM_HOME")
  if configured_root:
    return str(Path(configured_root).expanduser().resolve())

  source_root = Path(__file__).resolve().parents[2]
  if (source_root / "pyproject.toml").is_file() and (
    source_root / "seisLM"
  ).is_dir():
    return str(source_root)
  return str(Path.cwd())


def create_folder_if_not_exists(path):
  """
  This function checks if a folder exists at the given path.
  If it doesn't exist, it creates the folder.

  Args:
      path: The path to the folder to check and potentially create.
  """
  if not os.path.exists(path):
    try:
      os.makedirs(path)
      print(f"Folder created: {path}")
    except OSError as error:
      print(f"Error creating folder: {error}")

DATA_DIR = os.path.join(gitdir(), 'data')
MODEL_SAVE_DIR = os.path.join(gitdir(), 'results/models')
EVAL_SAVE_DIR = os.path.join(gitdir(), 'results/evaluation')
FIGURE_DIR = os.path.join(gitdir(), 'results/figures')


def list_all_checkpoints(
  base_folder: str,
  keywords: Optional[List[str]] = None
  ) -> List[str]:
  """List all checkpoints in the given folder, filtered by keywords.

  Args:
    base_folder: The folder to search for checkpoints.
    keywords: A list of keywords to filter the checkpoints.

  Returns:
    checkpoints: A list of checkpoint paths.
  """

  pattern = os.path.join(base_folder, '**', '*.ckpt')
  checkpoints = glob.glob(pattern, recursive=True)
  if keywords:
    checkpoints = [
        checkpoint for checkpoint in checkpoints
        if all(keyword in checkpoint for keyword in keywords)
    ]

  return checkpoints
