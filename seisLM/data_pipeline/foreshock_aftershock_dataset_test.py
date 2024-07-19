'''Test the updated foreshock_aftershock_dataset module.'''
from datetime import datetime
import numpy as np
import pandas as pd
from lightning import seed_everything
from seisLM.data_pipeline import foreshock_aftershock_dataset as myu
from seisLM.data_pipeline import ref_foreshock_aftershock_dataset as u
from seisLM.utils.project_path import DATA_DIR

dataToProcess = "NRCA"
seed = 42
path = f'{DATA_DIR}/foreshock_aftershock_NRCA/'


def convert_to_one_hot(array, num_classes):
  # Determine the number of unique classes
  num_classes = np.max(array) + 1

  # Convert to one-hot encoding
  one_hot_encoded = np.eye(num_classes)[array]
  return one_hot_encoded


def laurenti_preprocess(
  num_classes, seed, split_random):
  force_traces_in_test=[]
  seed_everything(seed)
  df_empty = pd.DataFrame(columns = [
      'E_channel', 'N_channel', 'Z_channel', 'trace_name', 'label',
      'trace_start_time', 'network_code', 'receiver_name', 'receiver_type',
      'receiver_elevation_m', 'receiver_latitude', 'receiver_longitude',
      'source_id', 'source_depth_km', 'source_latitude', 'source_longitude',
      'source_magnitude_type', 'source_magnitude', 'source_origin_time',
      'p_travel_sec']
  )
  df_pre = df_empty.copy()
  df_visso = df_empty.copy() # if num_classes!=9 this df will remain empty
  df_post = df_empty.copy()

  df_pre = pd.read_pickle(path+'dataframe_pre_'+dataToProcess+'.csv')
  df_post = pd.read_pickle(path+'dataframe_post_'+dataToProcess+'.csv')
  if num_classes==9:
    df_visso = pd.read_pickle(path+'dataframe_visso_'+dataToProcess+'.csv')

  df_pre, df_visso, df_post=u.pre_post_equal_length(
    df_pre, df_visso, df_post,force_traces_in_test, num_classes)

  for i in force_traces_in_test:
    if (i not in df_pre['trace_name'].values) and (
      i not in df_visso['trace_name'].values) and (
        i not in df_post['trace_name'].values):
      print("WARNING: ", i,
            " not in df_pre and df_post. This will cause an error.")

  df_pre['trace_start_time'] = df_pre['trace_start_time'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
  df_visso['trace_start_time'] = df_visso['trace_start_time'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
  df_post['trace_start_time'] = df_post['trace_start_time'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))


  if num_classes==2:
    df=pd.concat([df_pre, df_post], ignore_index=True)
  else:
    frames_pre = u.frames_N_classes(df_pre,num_classes, pre_or_post="pre")
    frames_post = u.frames_N_classes(df_post,num_classes, pre_or_post="post")
    if num_classes==9:
      frames_visso = u.frames_N_classes(
        df_visso,num_classes, pre_or_post="visso"
      )
      df=pd.concat(
        [pd.concat(frames_pre),
         pd.concat(frames_visso),
         pd.concat(frames_post)],
        ignore_index=True
      )
    else:
      df = pd.concat(
        [pd.concat(frames_pre),pd.concat(frames_post)], ignore_index=True
      )

  df['source_origin_time'] = df['source_origin_time'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
  df['TTF'] = df.apply (lambda row: u.add_TTF_in_sec(row), axis=1)

  (_, X_train, y_train, _, X_val, y_val, _, X_test, y_test, _
   ) = u.train_val_test_split(
    df.copy(),
    split_random=split_random,
  )
  return X_train, y_train, X_val, y_val, X_test, y_test

for split_random in [False, True]:
  for num_classes in [9, 4, 2]:
    print(f'split_random={split_random}, num_classes={num_classes}')
    datasets = myu.create_foreshock_aftershock_datasets(
      num_classes=num_classes,
      event_split_method='random' if split_random else 'temporal',
      component_order='ENZ',
      dimension_order='NWC',
      seed=seed,
    )

    train_data, val_data, test_data = datasets['train'], datasets['val'], datasets['test']

    X_train, y_train, X_val, y_val, X_test, y_test = laurenti_preprocess(
      num_classes=num_classes,
      seed=seed,
      split_random=split_random,
    )

    np.testing.assert_array_equal(X_train.astype(np.float32), train_data['X'])
    np.testing.assert_array_equal(
      y_train, convert_to_one_hot(train_data['y'], num_classes)
    )

    np.testing.assert_array_equal(X_val.astype(np.float32), val_data['X'])
    np.testing.assert_array_equal(
      y_val, convert_to_one_hot(val_data['y'], num_classes)
    )


    np.testing.assert_array_equal(X_test.astype(np.float32), test_data['X'])
    np.testing.assert_array_equal(
      y_test, convert_to_one_hot(test_data['y'],num_classes)
    )





