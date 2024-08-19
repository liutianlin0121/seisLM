import os

# Define arrays for data names and training fractions
configs = [
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/ethz_phasenet.json',
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/geofon_phasenet.json',
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/ethz_eqtransformer.json',
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/geofon_eqtransformer.json',
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/ethz_gpdpick.json',
    '/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/geofon_gpssdpick.json',
]

training_fractions = [0.05, 0.1, 0.2, 0.5, 1.0]

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

# Iterate over each combination of config and training fraction
for config in configs:
  for training_fraction in training_fractions:
    # Construct the command
    command = f"python3 phasepick_run.py --config {config} --training_fraction {training_fraction} --save_checkpoints"
    # Run the command
    os.system(command)
