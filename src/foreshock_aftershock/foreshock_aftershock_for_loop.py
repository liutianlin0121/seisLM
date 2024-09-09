import os

training_fractions = [0.05, 0.1, 0.2, 0.5, 1.0]


for num_classes in [9, 4]:
  # Iterate over each combination of config and training fraction
  for training_fraction in training_fractions:
    # Construct the command
    command = f"python3 foreshock_aftershock_run.py --config /home/liu0003/Desktop/projects/seisLM/seisLM/configs/foreshock_aftershock/seisLM_shock_classifier.json --num_classes {num_classes} --training_fraction {training_fraction} --save_checkpoints"
    # Run the command
    os.system(command)
