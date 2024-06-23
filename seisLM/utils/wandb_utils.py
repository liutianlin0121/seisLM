"""Utils for wandb"""
import os
import threading
import time
import psutil


# End the WandB run
# wandb.finish()
# FIXME: workaround to prevent wandb from blocking the termination
# of runs on sciCORE slurm
def aux(pid, timeout=60):
  time.sleep(timeout)
  print("Program did not terminate successfully, killing process tree")
  parent = psutil.Process(pid)
  for child in parent.children(recursive=True):
    child.kill()
  parent.kill()

shutdown_cleanup_thread = threading.Thread(
  target=aux, args=(os.getpid(), 60), daemon=True)
