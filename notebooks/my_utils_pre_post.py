'''Utility functions for the foreshock aftershock dataset'''
from datetime import datetime
import pandas as pd
from lightning import seed_everything


def equallize_dataset_length(
  df_pre, df_visso, df_post, num_classes, seed=42):

  seed_everything(seed)

  # Randomly shuffled rows and reset their indices
  df_pre=df_pre.sample(frac=1).reset_index(drop=True)
  if isinstance(df_visso, pd.DataFrame):
    df_visso=df_visso.sample(frac=1).reset_index(drop=True)
  df_post=df_post.sample(frac=1).reset_index(drop=True)

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


def compute_norcia_ttf(row):
  """It takes a row from a df and it computes the difference in seconds between
  the event in the input row and the main event.
  This is called Time To Failure (TTF)
  Args:
  ----------
  row : pandas.core.series.Series
        row from Input DataFrame where to add column TTF
  Returns:
  ----------
  difference : float of the amount of time in seconds between the event
  in the input row and the main one
  """
  time=row['source_origin_time']
  norcia_datetime= datetime.strptime(
    '2016-10-30T07:40:17.000000Z', '%Y-%m-%dT%H:%M:%S.%fZ'
  )
  difference = (time-norcia_datetime).total_seconds()
  return difference
