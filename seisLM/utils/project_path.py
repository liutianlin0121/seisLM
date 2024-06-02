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
