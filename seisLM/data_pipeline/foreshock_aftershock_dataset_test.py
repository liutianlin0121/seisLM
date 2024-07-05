'''Test the updated foreshock_aftershock_dataset module.'''
from datetime import datetime
import numpy as np
import pandas as pd
from lightning import seed_everything
from seisLM.data_pipeline import foreshock_aftershock_dataset as myu
from seisLM.data_pipeline import ref_foreshock_aftershock_dataset as u

dataToProcess = "NRCA"
seed = 42
path = '/scicore/home/dokman0000/liu0003/projects/seisLM/data/wetransfer_classify_generic_norcia-py_2024-06-24_1530/'

train_percentage = 0.7
val_percentage = 0.10
test_percentatge = 0.20


def laurenti_preprocess(num_classes, train_frac, val_frac, test_frac, seed):
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
    split_random=True,
  )
  return X_train, y_train, X_val, y_val, X_test, y_test

for num_classes in [2, 4, 9]:
  datasets = myu.create_foreshock_aftershock_datasets(
    num_classes=num_classes,
    train_frac=train_percentage,
    val_frac=val_percentage,
    test_frac=test_percentatge,
    seed=seed
  )

  train_data, val_data, test_data = datasets['train'], datasets['val'], datasets['test']

  X_train, y_train, X_val, y_val, X_test, y_test = laurenti_preprocess(
    num_classes=num_classes,
    train_frac=train_percentage,
    val_frac=val_percentage,
    test_frac=test_percentatge,
    seed=seed
  )

  np.testing.assert_array_equal(X_train, train_data['X'])
  np.testing.assert_array_equal(y_train, train_data['y'])

  np.testing.assert_array_equal(X_val, val_data['X'])
  np.testing.assert_array_equal(y_val, val_data['y'])


  np.testing.assert_array_equal(X_test, test_data['X'])
  np.testing.assert_array_equal(y_test, test_data['y'])





