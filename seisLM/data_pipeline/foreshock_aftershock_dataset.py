"""Utility functions for the foreshock aftershock dataset"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import einops
import numpy as np
import pandas as pd
from lightning import seed_everything
from sklearn.utils import shuffle

from seisLM.utils.project_path import DATA_DIR


def compute_norcia_ttf(row: pd.core.series.Series) -> float:
  """Computes the difference in seconds between an event the main event.
  This difference in time is called Time To Failure (TTF).

  Args:
    row : pandas.core.series.Series
        row from Input DataFrame where to add column TTF

  Returns:
    difference : float of the amount of time in seconds between the event
    in the input row and the main one
  """
  time = row["source_origin_time"]
  norcia_datetime = datetime.strptime(
    "2016-10-30T07:40:17.000000Z", "%Y-%m-%dT%H:%M:%S.%fZ"
  )
  difference = (time - norcia_datetime).total_seconds()
  return difference


def shuffle_and_reset(
  df: pd.DataFrame, drop: bool = False, seed: int = None
) -> pd.DataFrame:
  """Shuffle the DataFrame and reset the index."""
  df = shuffle(df, random_state=seed)
  df.reset_index(inplace=True, drop=drop)
  return df


def equallize_dataset_length(
  df_pre: pd.DataFrame,
  df_visso: Optional[pd.DataFrame],
  df_post: pd.DataFrame,
  num_classes: int,
  seed: int = 42,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
  """Truncate the DataFrames to make sure each class has the same length."""
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
    trunc_length_visso_class = (
      min(len(df_pre), len(df_post), len(df_visso) * num_non_visso_classes)
      // num_non_visso_classes
    )
    trunc_length_pre_and_post = trunc_length_visso_class * num_non_visso_classes

    # Truncate df_visso to the appropriate length
    df_visso = df_visso[:trunc_length_visso_class]

  df_pre = df_pre[:trunc_length_pre_and_post]
  df_post = df_post[:trunc_length_pre_and_post]

  return df_pre, df_visso, df_post


def equip_shock_dfs_with_class_labels(
  df_pre: pd.DataFrame,
  df_visso: Optional[pd.DataFrame],
  df_post: pd.DataFrame,
  num_classes: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
  """
  Takes a DataFrame and returns a list of sub-DataFrames with new labels.

  Args:
    df_pre : DataFrame of pre-shock events
    df_visso : Optional DataFrame or visso-shock events
    df_post : DataFrame of post-shock events
    num_classes : Number of total classes to split the df into.

  Returns:
    Tuple of DataFrames with class labels.
  """

  if num_classes % 2 == 1:
    # When the number of classes is odd, the visso class must be included.
    assert isinstance(df_visso, pd.DataFrame)
    df_visso.sort_values(by="trace_start_time", inplace=True)
    df_visso.reset_index(inplace=True)

    # Create a label DataFrame with the same number of rows
    labels = [num_classes // 2] * len(df_visso)

    # Assign the label DataFrame to the visso_frame
    df_visso = df_visso.assign(label=labels)

  num_pre_or_post_classes = num_classes // 2
  post_idx_shift = num_classes % 2

  def process_df(df: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    df.sort_values(by="trace_start_time", inplace=True)
    rows_per_class = len(df) // num_pre_or_post_classes
    frames = []
    for c in range(num_pre_or_post_classes):
      frame = df.iloc[
        c * rows_per_class : (c + 1) * rows_per_class
      ].reset_index(drop=True)
      label_position = c + offset
      frame["label"] = [label_position] * len(frame)
      frames.append(frame)
    return pd.concat(frames, ignore_index=True)

  df_pre = process_df(df_pre)
  df_post = process_df(df_post, num_pre_or_post_classes + post_idx_shift)
  return df_pre, df_visso, df_post


def extract_input_target_from_dataframe(
  *,
  df: pd.DataFrame,
  component_order: str,
  dimension_order: str,
) -> Dict[str, Union[np.ndarray, str]]:
  """Extract the input values and the target values from a DataFrame."""

  # Create a mapping of channels based on the component order
  channel_mapping = {"E": "E_channel", "N": "N_channel", "Z": "Z_channel"}

  assert dimension_order in ["NCW", "NWC"]

  # Ensure the component_order is valid
  assert set(component_order) == {
    "E",
    "N",
    "Z",
  }, "component_order must be a permutation of 'ENZ'"

  # Generate the input_values based on the component order
  input_values = np.array(
    df.apply(
      lambda row: np.stack(
        [row[channel_mapping[comp]] for comp in component_order]
      ),
      axis=1,
    ).to_list(),
    dtype=np.float32,
  )

  if dimension_order == "NWC":
    input_values = einops.rearrange(input_values, "n c w -> n w c")
  targets = np.array(df["label"].to_list())

  occurence_time = (
    df["source_origin_time"]
    .apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"))
    .to_list()
  )

  return {"X": input_values, "y": targets, "occurence_time": occurence_time}


def train_val_test_split(
  *,
  df: pd.DataFrame,
  num_classes: int,
  component_order: str,
  dimension_order: str,
  train_frac: float = 0.70,
  val_frac: float = 0.10,
  test_frac: float = 0.20,
  event_split_method: str = "random",
  seed: int = 42,
  verbose_events: bool = False,
) -> Tuple[
  Dict[str, Union[np.ndarray, str]],
  Dict[str, Union[np.ndarray, str]],
  Dict[str, Union[np.ndarray, str]],
]:
  """
  Split the dataset in train, val, and test folds.

  In the splitting, we make sure that the events in the different folds
  come from different events. That is, the split is temporal rather than
  random. It is to avoid the possibility of model shortcuts in time content.
  """

  assert train_frac + val_frac + test_frac <= 1, (
    "The sum of train_frac, val_frac, and test_frac must be less than or "
    "equal to 1."
  )

  seed_everything(seed)

  if event_split_method == "random":
    # randomly assign events to train, val, and test

    # source_id are the identifications events;
    # they distinguish one earthquake from another;
    source_id_array = df["source_id"].unique()
    source_id_array.sort()

    source_id_array = np.array(source_id_array)

    np.random.shuffle(source_id_array)

    train_end = int(len(source_id_array) * train_frac)
    val_end = int(len(source_id_array) * (train_frac + val_frac))

    source_id_train = source_id_array[:train_end]
    source_id_val = source_id_array[train_end:val_end]
    source_id_test = source_id_array[val_end:]

  elif event_split_method == "temporal":
    # temporally assign events to train, val, and test.

    frames_class = [df[df["label"] == i] for i in range(num_classes)]

    source_id_train = []
    source_id_val = []
    source_id_test = []

    for df_frame in frames_class:
      df_frame = df_frame.sort_values(by=["trace_start_time"])
      source_id_array = df_frame["source_id"].unique()
      n_traces_train = int(len(source_id_array) * train_frac)
      n_traces_val = int(len(source_id_array) * val_frac)
      n_traces_test = int(len(source_id_array) * test_frac)

      train_split = int(n_traces_train / 2)
      val_split = train_split + n_traces_val
      test_split = val_split + n_traces_test

      source_id_train_frame = np.concatenate(
        (source_id_array[:train_split], source_id_array[-train_split:]), axis=0
      )
      source_id_val_frame = source_id_array[train_split:val_split]
      source_id_test_frame = source_id_array[val_split:test_split]

      source_id_train.extend(source_id_train_frame.astype(int))
      source_id_val.extend(source_id_val_frame.astype(int))
      source_id_test.extend(source_id_test_frame.astype(int))

  else:
    raise ValueError("event_split_method must be 'random' or 'temporal'")

  if verbose_events:
    print("Events in train dataset: ", len(source_id_train))
    print("Events in validation dataset: ", len(source_id_val))
    print("Events in test dataset: ", len(source_id_test))

  train_df = shuffle_and_reset(df.loc[df["source_id"].isin(source_id_train)])
  val_df = shuffle_and_reset(df.loc[df["source_id"].isin(source_id_val)])
  test_df = shuffle_and_reset(df.loc[df["source_id"].isin(source_id_test)])

  train_data = extract_input_target_from_dataframe(
    df=train_df,
    component_order=component_order,
    dimension_order=dimension_order,
  )
  val_data = extract_input_target_from_dataframe(
    df=val_df,
    component_order=component_order,
    dimension_order=dimension_order,
  )
  test_data = extract_input_target_from_dataframe(
    df=test_df,
    component_order=component_order,
    dimension_order=dimension_order,
  )

  return train_data, val_data, test_data


def create_foreshock_aftershock_datasets(
  *,
  num_classes: int,
  event_split_method: str,
  component_order: str,
  dimension_order: str,
  remove_class_overlapping_dates: bool = False,
  train_frac: float = 0.70,
  val_frac: float = 0.10,
  test_frac: float = 0.20,
  seed: int = 42,
) -> Dict[str, Dict[str, Union[np.ndarray, str]]]:
  df_pre = pd.read_pickle(
    f"{DATA_DIR}/foreshock_aftershock_NRCA/dataframe_pre_NRCA.csv"
  )
  df_post = pd.read_pickle(
    f"{DATA_DIR}/foreshock_aftershock_NRCA/dataframe_post_NRCA.csv"
  )

  if num_classes % 2 == 1:
    df_visso = pd.read_pickle(
      f"{DATA_DIR}/foreshock_aftershock_NRCA/dataframe_visso_NRCA.csv"
    )
  else:
    df_visso = None

  df_pre, df_visso, df_post = equallize_dataset_length(
    df_pre, df_visso, df_post, num_classes, seed=seed
  )

  if num_classes == 2:
    df = pd.concat([df_pre, df_post], ignore_index=True)
    df["label"] = df["label"].apply(lambda x: x.index(max(x)))
  else:
    df_tuple_pre_visso_post = equip_shock_dfs_with_class_labels(
      df_pre, df_visso, df_post, num_classes
    )
    df = pd.concat(
      [df for df in df_tuple_pre_visso_post if df is not None],
      ignore_index=True,
    )

  train_data, val_data, test_data = train_val_test_split(
    df=df,
    num_classes=num_classes,
    component_order=component_order,
    dimension_order=dimension_order,
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac,
    event_split_method=event_split_method,
    seed=seed,
  )

  if remove_class_overlapping_dates:
    train_data = remove_dates_shared_by_classes(train_data)
    val_data = remove_dates_shared_by_classes(val_data)
    test_data = remove_dates_shared_by_classes(test_data)

  datasets = {"train": train_data, "val": val_data, "test": test_data}
  return datasets


def remove_dates_shared_by_classes(
  data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
  """Remove dates that exist in multiple classes from the dataset."""

  # For each date, record the classes that appear on that date.
  date_to_classes = defaultdict(set)
  for timestamp, label in zip(data["occurence_time"], data["y"]):
    date_to_classes[timestamp.date()].add(label)

    # A single date cannot be in more than two classes;
    # if it is, the dataset is prepared incorrectly.
    assert len(date_to_classes[timestamp.date()]) <= 2

  # Identify dates that appear in more than one class
  dates_in_multiple_classes = {
    date for date, classes in date_to_classes.items() if len(classes) > 1
  }

  # Filter out entries with dates in multiple classes
  filtered_X = []
  filtered_y = []
  filtered_occurence_time = []

  for x, y, timestamp in zip(data["X"], data["y"], data["occurence_time"]):
    if timestamp.date() not in dates_in_multiple_classes:
      filtered_X.append(x)
      filtered_y.append(y)
      filtered_occurence_time.append(timestamp)

  filtered_X = np.array(filtered_X)
  filtered_y = np.array(filtered_y)

  data["X"] = filtered_X
  data["y"] = filtered_y
  data["occurence_time"] = filtered_occurence_time
  return data
