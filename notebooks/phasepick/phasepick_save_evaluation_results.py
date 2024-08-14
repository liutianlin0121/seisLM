""" Save evaluation results for phase picking models. """
from seisLM.evaluation import pick_eval
from seisLM.utils import project_path
from phasepick_model_registry import (
  # phasenet_ethz,
  # phasenet_geofon,
  # seislm_ethz,
  seislm_geofon,
)


all_ckpt_dicts = {
  # ('ethz', 'MultiDimWav2Vec2ForFrameClassification'): seislm_ethz,
  ('geofon', 'MultiDimWav2Vec2ForFrameClassification'): seislm_geofon,
  # ('ethz', 'PhaseNet'): phasenet_ethz,
  # ('geofon', 'PhaseNet'): phasenet_geofon,
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
