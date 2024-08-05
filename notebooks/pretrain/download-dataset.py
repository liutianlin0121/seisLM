from seisLM.data_pipeline import seisbench_dataloaders as dataloaders

dataset_names = [
  "ETHZ",
  "GEOFON",
  "STEAD",
  "NEIC",
  "InstanceCounts",
  "Iquique",
]


for dataset_name in dataset_names:
  print(dataset_name)
  dataset = dataloaders.get_dataset_by_name(dataset_name)(
      sampling_rate=100,
      component_order='ZNE',
      dimension_order='NCW',
      cache=None
  )
