'''Utility functions for the foreshock aftershock dataset'''
from datetime import datetime
from typing import Optional

import einops
import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from sklearn.utils import shuffle


def compute_norcia_ttf(row):
  """Computes the difference in seconds between an event the main event.
  This difference in time is called Time To Failure (TTF).

  Args:
    row : pandas.core.series.Series
        row from Input DataFrame where to add column TTF

  Returns:
    difference : float of the amount of time in seconds between the event
    in the input row and the main one
  """
  time=row['source_origin_time']
  norcia_datetime= datetime.strptime(
    '2016-10-30T07:40:17.000000Z', '%Y-%m-%dT%H:%M:%S.%fZ'
  )
  difference = (time-norcia_datetime).total_seconds()
  return difference

def shuffle_and_reset(
  df: pd.DataFrame, drop: bool = False, seed: int = None
  ) -> pd.DataFrame:
  """ Shuffle the DataFrame and reset the index."""
  df = shuffle(df, random_state=seed)
  df.reset_index(inplace=True, drop=drop)
  return df


def equallize_dataset_length(
  df_pre: pd.DataFrame,
  df_visso: Optional[pd.DataFrame],
  df_post: pd.DataFrame,
  num_classes: int,
  seed: int = 42
  ):

  """ Truncate the DataFrames to make sure each class has the same length."""
  seed_everything(seed)

  # Randomly shuffled rows and reset their indices
  df_pre = shuffle_and_reset(df_pre, drop=True)
  if isinstance(df_visso, pd.DataFrame):
    df_visso = shuffle_and_reset(df_visso, drop=True)
  df_post = shuffle_and_reset(df_post, drop=True)

  if not isinstance(df_visso, pd.DataFrame):
    # no visso class
    # Truncate the dataframes to the length of the shortest one
    trunc_length_pre_and_post = min(len(df_pre), len(df_post))
  else:
    # if visso dataset exists, then the number of classes must be odd,
    # because we want to evently split the classes between pre and post.
    assert num_classes % 2 == 1
    num_non_visso_classes = num_classes // 2

    # Determine the length class
    trunc_length_visso_class = min(
      len(df_pre), len(df_post), len(df_visso) * num_non_visso_classes
    ) // num_non_visso_classes
    trunc_length_pre_and_post = (
      trunc_length_visso_class * num_non_visso_classes
    )

    # Truncate df_visso to the appropriate length
    df_visso = df_visso[:trunc_length_visso_class]

  df_pre=df_pre[:trunc_length_pre_and_post]
  df_post=df_post[:trunc_length_pre_and_post]

  return df_pre, df_visso, df_post


def split_df_into_class_dependent_frames(df, num_classes, pre_or_post):
  """
  Takes a DataFrame and returns a list of sub-DataFrames with new labels.

  Args:
  ----------
  df : DataFrame
      Input DataFrame (pre, post, or visso) from where to recompute classes
  num_classes : int
      Number of total classes to split the df into (pre, post, and eventually
      visso if num_classes == 9)
  pre_or_post : str
      "pre", "post", or "visso" to properly assign the new label

  Returns:
  ----------
  frames : list of sub-DataFrames from the original df
  """

  # Rename the label column and sort the DataFrame by 'trace_start_time'
  df = df.rename(columns={'label': 'label_2classes'})
  df.sort_values(by='trace_start_time', inplace=True)

  # Function to create a label array with a 1 at the specified position
  def create_label_array(num_classes, position):
    label_array = [0] * num_classes
    label_array[position] = 1
    return label_array

  frames = []

  if pre_or_post == "visso":
    # Case for 'visso'
    # Copy the entire DataFrame into frames
    visso_frame = df.copy()
    visso_frame.reset_index(inplace=True)

    # Create a label DataFrame with the same number of rows
    labels = [create_label_array(num_classes, num_classes // 2)] * len(
        visso_frame)

    # Assign the label DataFrame to the visso_frame
    visso_frame = visso_frame.assign(label=labels)

    # Append the visso_frame to the frames list
    frames.append(visso_frame)

  elif pre_or_post in ["pre", "post"]:
    # Case for 'pre' and 'post'
    # Determine the number of rows per sub-DataFrame

    num_pre_or_post_classes = num_classes // 2
    rows_per_class = len(df) // num_pre_or_post_classes
    post_idx_shift = num_classes % 2

    # Split the DataFrame into equal parts and process each part
    for c in range(num_pre_or_post_classes):
      frame = df.iloc[c * rows_per_class: (c + 1) * rows_per_class].copy()
      frame.reset_index(inplace=True)

      labels = []
      for _ in range(len(frame)):
        if pre_or_post == "pre":
          label_array = create_label_array(num_classes, c)
        elif pre_or_post == "post":
          label_array = create_label_array(
            num_classes, num_pre_or_post_classes + c + post_idx_shift
          )
        labels.append(label_array)
      frame = frame.assign(label=labels)
      frames.append(frame)
  else:
    raise ValueError("pre_or_post must be 'pre', 'visso', or 'post'")
  return frames



def extract_input_target_from_dataframe(df):

  input_values = np.array(
    df.apply(lambda row: np.stack(
      [row['E_channel'], row['N_channel'], row['Z_channel']]), axis=1
    ).to_list()
  )
  input_values = einops.rearrange(input_values, 'b c l -> b l c')
  targets = np.array(df['label'].to_list())

  return {'X': input_values, 'y': targets}



def train_val_test_split(
  df, train_frac=0.70, val_frac=0.10, test_frac=0.20,
  seed=42, verbose_events=False):

  """
  Split the dataset in train, val, and test folds.

  In the splitting, we make sure that the events in the different folds
  come from different events. That is, the split is temporal rather than
  random. It is to avoid the possibility of model shortcuts in time content.
  """

  seed_everything(seed)

  # source_id are the identifications events;
  # they distinguish one earthquake from another;
  source_id_array = df['source_id'].unique()
  source_id_array.sort()

  source_id_array = np.array(source_id_array)

  np.random.shuffle(source_id_array)

  train_end = int(len(source_id_array) * train_frac)
  val_end = int(len(source_id_array) * (train_frac + val_frac))

  source_id_train = source_id_array[:train_end]
  source_id_val = source_id_array[train_end:val_end]
  source_id_test = source_id_array[val_end:]

  if verbose_events:
    print("Events in train dataset: ",len(source_id_train))
    print("Events in validation dataset: ",len(source_id_val))
    print("Events in test dataset: ",len(source_id_test))

  # Apply the function to each subset
  train_df = shuffle_and_reset(df.loc[df['source_id'].isin(source_id_train)])
  val_df = shuffle_and_reset(df.loc[df['source_id'].isin(source_id_val)])
  test_df = shuffle_and_reset(df.loc[df['source_id'].isin(source_id_test)])

  train_data = extract_input_target_from_dataframe(train_df)
  val_data = extract_input_target_from_dataframe(val_df)
  test_data = extract_input_target_from_dataframe(test_df)

  return train_data, val_data, test_data



def create_foreshock_aftershock_datasets(
  num_classes,
  data_path='/scicore/home/dokman0000/liu0003/projects/seisLM/data/wetransfer_classify_generic_norcia-py_2024-06-24_1530/',
  station='NRCA',
  train_frac=0.70,
  val_frac=0.10,
  test_frac=0.20,
  seed=42
  ):

  df_pre = pd.read_pickle(data_path + 'dataframe_pre_'+ station + '.csv')
  df_post = pd.read_pickle(data_path + 'dataframe_post_'+ station +'.csv')

  if num_classes % 2 == 1:
    df_visso = pd.read_pickle(data_path+'dataframe_visso_'+ station +'.csv')
  else:
    df_visso = None

  df_pre, df_visso, df_post = equallize_dataset_length(
    df_pre, df_visso, df_post, num_classes, seed=seed
  )

  if num_classes==2:
    df = pd.concat([df_pre, df_post], ignore_index=True)
  else:
    frames_pre = split_df_into_class_dependent_frames(
      df_pre, num_classes, pre_or_post="pre"
    )
    frames_post = split_df_into_class_dependent_frames(
      df_post, num_classes, pre_or_post="post"
    )
    if isinstance(df_visso, pd.DataFrame):
      frames_visso = split_df_into_class_dependent_frames(
        df_visso, num_classes, pre_or_post="visso"
      )
      df=pd.concat(
        [pd.concat(frames_pre),pd.concat(frames_visso),pd.concat(frames_post)],
        ignore_index=True
      )
    else:
      df=pd.concat(
        [pd.concat(frames_pre),pd.concat(frames_post)], ignore_index=True
      )

  train_data, val_data, test_data = train_val_test_split(
    df.copy(),
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac,
    seed=seed
  )

  datasets = {
    'train': train_data,
    'val': val_data,
    'test': test_data
  }
  return datasets


def create_foreshock_aftershock_dataloaders(
  num_classes,
  batch_size,
  data_path='/scicore/home/dokman0000/liu0003/projects/seisLM/data/wetransfer_classify_generic_norcia-py_2024-06-24_1530/',
  station='NRCA',
  seed=42,
  train_frac=0.70,
  val_frac=0.10,
  test_frac=0.20,
  standardize=True,
  num_workers=8,
  ):

  datasets = create_foreshock_aftershock_datasets(
    num_classes=num_classes,
    data_path=data_path,
    station=station,
    seed=seed,
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac
  )

  X_train, y_train = datasets['train']['X'], datasets['train']['y']
  X_val, y_val = datasets['val']['X'], datasets['val']['y']
  X_test, y_test = datasets['test']['X'], datasets['test']['y']

  if standardize:
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

  X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
  X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
  X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

  loaders = {
    'train': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_train, y_train),
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
    ),
    'val': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_val, y_val),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
    ),
    'test': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_test, y_test),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
    )
  }

  return loaders
