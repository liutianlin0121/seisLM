"""Utility functions that provide the absolute path to project directories.

Typical usage example:

# Path to the checkpoint directory used to store intermediate training
# checkpoints for experiment name stored in `experiment_name`.
checkpointsdir(experiment_name)
"""

import os
import git


def gitdir() -> str:
  """Find the absolute path to the GitHub repository root.
  """
  git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
  git_root = git_repo.git.rev_parse('--show-toplevel')
  return git_root

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

