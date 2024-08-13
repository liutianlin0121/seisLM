""" Save evaluation results for phase picking models. """
from seisLM.evaluation import pick_eval
from seisLM.utils import project_path

# %%
phasenet_ethz = {
  '0.05': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_ETHZ_train_frac_0.05_model_PhaseNet_seed_42_time_2024-08-11-16h-57m-27s/checkpoints/epoch=49-step=800.ckpt",
  '0.1': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_ETHZ_train_frac_0.1_model_PhaseNet_seed_42_time_2024-08-11-16h-59m-42s/checkpoints/epoch=49-step=2400.ckpt",
  '0.2': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_ETHZ_train_frac_0.2_model_PhaseNet_seed_42_time_2024-08-11-17h-01m-13s/checkpoints/epoch=48-step=3920.ckpt",
  '0.5': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_ETHZ_train_frac_0.5_model_PhaseNet_seed_42_time_2024-08-11-17h-03m-46s/checkpoints/epoch=37-step=6498.ckpt",
  '1.0': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_ETHZ_train_frac_1.0_model_PhaseNet_seed_42_time_2024-08-11-17h-04m-13s/checkpoints/epoch=37-step=13414.ckpt",
}


phasenet_geofon = {
  '0.05': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_GEOFON_train_frac_0.05_model_PhaseNet_seed_42_time_2024-08-11-16h-47m-30s/checkpoints/epoch=41-step=5208.ckpt",
  '0.1': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_GEOFON_train_frac_0.1_model_PhaseNet_seed_42_time_2024-08-11-16h-47m-30s/checkpoints/epoch=42-step=10965.ckpt",
  '0.2': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_GEOFON_train_frac_0.2_model_PhaseNet_seed_42_time_2024-08-11-16h-47m-30s/checkpoints/epoch=26-step=13392.ckpt",
  '0.5': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_GEOFON_train_frac_0.5_model_PhaseNet_seed_42_time_2024-08-11-16h-47m-31s/checkpoints/epoch=28-step=36801.ckpt",
  '1.0': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/phasenet_GEOFON_train_frac_1.0_model_PhaseNet_seed_42_time_2024-08-11-16h-47m-31s/checkpoints/epoch=20-step=52920.ckpt",
}


seislm_ethz = {
  '0.05': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_ETHZ_train_frac_0.05_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-20h-00m-41s/checkpoints/epoch=40-step=656.ckpt", 
  '0.1': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_ETHZ_train_frac_0.1_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-20h-06m-11s/checkpoints/epoch=41-step=2016.ckpt",
  '0.2': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_ETHZ_train_frac_0.2_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-20h-14m-40s/checkpoints/epoch=44-step=3600.ckpt",
  '0.5': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_ETHZ_train_frac_0.5_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-20h-26m-11s/checkpoints/epoch=36-step=6327.ckpt",
  '1.0': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_ETHZ_train_frac_1.0_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-20h-46m-05s/checkpoints/epoch=46-step=16591.ckpt"
}

seislm_geofon = {
  '0.05': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_GEOFON_train_frac_0.05_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-18h-36m-00s/checkpoints/epoch=38-step=4836.ckpt",
  '0.1': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_GEOFON_train_frac_0.1_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-18h-36m-01s/checkpoints/epoch=46-step=11985.ckpt",
  '0.2': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_GEOFON_train_frac_0.2_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-19h-01m-20s/checkpoints/epoch=49-step=24800.ckpt",
  '0.5': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_GEOFON_train_frac_0.5_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-19h-13m-26s/checkpoints/epoch=48-step=62181.ckpt",
  '1.0': "/home/liu0003/Desktop/projects/seisLM/results/models/phasepick_run/seisLM_convpos_GEOFON_train_frac_1.0_model_MultiDimWav2Vec2ForFrameClassification_seed_42_time_2024-08-12-19h-25m-23s/checkpoints/epoch=36-step=93240.ckpt",
}

all_ckpt_dicts = {
  # ('ethz', 'MultiDimWav2Vec2ForFrameClassification'): seislm_ethz,
  # ('geofon', 'MultiDimWav2Vec2ForFrameClassification'): seislm_geofon,
  ('ethz', 'PhaseNet'): phasenet_ethz,
  ('geofon', 'PhaseNet'): phasenet_geofon,
}


# sets = 'dev,test'
sets = 'test'

for (dataset, model_name), ckpt_dict in all_ckpt_dicts.items():
  for frac, ckpt in ckpt_dict.items():
    print(f"dataset: {dataset}, model: {model_name}, frac: {frac}")
    print(f"ckpt: {ckpt}")
    save_tag = ckpt.split('/')[-3]
    print(f'model tag, {save_tag}')
    pick_eval.save_pick_predictions(
        checkpoint_path_or_data_name=ckpt,
        save_tag=ckpt.split('/')[-3],
        model_name=model_name,
        targets=project_path.gitdir() + f'/data/targets/{dataset}/',
        sets=sets,
        batchsize=64
    )
