import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# import scipy
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# from tqdm.auto import tqdm
import random
import os
# import copy
import math
# import gc
# import h5py
# from obspy.imaging.spectrogram import spectrogram


# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
#     dev = "cpu"
# device = torch.device(dev)

def seed_everything(seed):
    """It sets all the seeds for reproducibility.

    Args:
    ----------
    seed : int
        Seed for all the methods
    """
    print("Setting seeds")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def pre_post_equal_length(df_pre, df_visso, df_post, force_in_test, num_classes):
    """ This function is to ensure to have a balanced data. We remove events 
    from the bigger dataset, in order to have the same number of events pre and post Norcia.
    To make sure to remove random events in time, it first shuffles the rows
    Args:
    ----------
    df_pre : DataFrame
        Input DataFrame pre from where to eventually remove to make the dataset balanced
    df_visso : DataFrame
        Input DataFrame visso from where to eventually remove to make the dataset balanced. It this is empty it won't count.
    df_post : DataFrame
        Input DataFrame post from where to eventually remove to make the dataset balanced
    force_in_test : list
        traces to be forced in test set
    num_classes : int
        number of total classes we eant to split the df (pre, post and eventually visso, if num_classes==9)

    Returns:
    ----------
    df_pre : resulting DataFrame that is the shuffled version of the original one, and have the same number of events as df_post
    df_post : resulting DataFrame that is the shuffled version of the original one, and have the same number of events as df_pre
    """
    df_pre=df_pre.sample(frac=1).reset_index(drop=True)
    df_visso=df_visso.sample(frac=1).reset_index(drop=True)
    df_post=df_post.sample(frac=1).reset_index(drop=True)

    # check if traces in force_in_test belong to current station
    for i in force_in_test:
        trace_station=i.split('.')[0]
        df_station=df_pre['trace_name'][0].split('.')[0]
        if trace_station==df_station: # trace i in force_in_test belong to current station
            if i in df_pre['trace_name'].values: # trace i is in pre
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_pre.loc[df_pre['trace_name'] == i].index[0]
                idx = df_pre.index.tolist()
                idx.remove(index_to_shift)
                df_pre = df_pre.reindex([index_to_shift] + idx)
            elif i in df_visso['trace_name'].values: # trace i is in visso
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_visso.loc[df_visso['trace_name'] == i].index[0]
                idx = df_visso.index.tolist()
                idx.remove(index_to_shift)
                df_visso = df_visso.reindex([index_to_shift] + idx)
            else: # trace i is in post       
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_post.loc[df_post['trace_name'] == i].index[0]
                idx = df_post.index.tolist()
                idx.remove(index_to_shift)
                df_post = df_post.reindex([index_to_shift] + idx) 
        # else: trace i doesn't belong to current station, so cut df without caring

    # compute length 
    len_class=0
    if num_classes==9: # this includes visso
        if len(df_visso)*4>len(df_pre) or len(df_visso)*4>len(df_post):
            if len(df_pre)<len(df_post):
                len_class=int(len(df_pre)/4)*4
            else:
                len_class=int(len(df_post)/4)*4
        else:
            len_class=len(df_visso)*4             
        df_visso=df_visso[:int(len_class/4)] 
    else: # this doesn't include visso
        if len(df_pre)<len(df_post):
            len_class=len(df_pre)
        else:
            len_class=len(df_post)
    df_pre=df_pre[:len_class] 
    df_post=df_post[:len_class]

    return df_pre, df_visso, df_post

def frames_N_classes(df,num_classes, pre_or_post):
    """It takes a df and it returns a list of (int(num_classes/2)) sub-DataFrames from the original df
    Args:
    ----------
    df : DataFrame
        Input DataFrame pre or post from where to recompute classes
    num_classes : int
        number of total classes we eant to split the df (pre, post and eventually visso, if num_classes==9)
    pre_or_post : String
        "pre" or "post" or "visso". It's used to properly assign the new label
    Returns:
    ----------
    frames : list of (int(num_classes/2)) sub-DataFrames from the original df
        
    """

    df = df.rename(columns={'label': 'label_2classes'})
    df.sort_values(by='trace_start_time', inplace=True)
    if pre_or_post=="visso":
        N = len(df)
        frames = [ df.iloc[i*N:(i+1)*N].copy() for i in range(1)]
        for f in range(0,len(frames)):
            frames[f] = frames[f].reset_index()
            label = pd.DataFrame(columns=['label'])
            for i in range(0,len(frames[f])):
                lab= [0] * num_classes # initialize label as a 0 array
                lab[int(num_classes/2)]=1 
                label.at[int(i), "label"] = lab
            frames[f] = frames[f].assign(label=label)
    elif pre_or_post=="pre" or pre_or_post=="post":
      N = int(len(df)/int(num_classes/2))
      frames = [ df.iloc[i*N:(i+1)*N].copy() for i in range(int(num_classes/2))]
      for f in range(0,len(frames)):
            frames[f] = frames[f].reset_index()
            label = pd.DataFrame(columns=['label'])
            for i in range(0,len(frames[f])):
                  lab= [0] * num_classes # initialize label as a 0 array
                  if pre_or_post=="pre": # assign 1 to the correct class
                        lab[f]=1 
                  elif pre_or_post=="post":
                        if num_classes==9:
                              lab[int(num_classes/2)+f+1]=1 # let's shift by 1 to leave the place for "visso" class
                        else:
                              lab[int(num_classes/2)+f]=1
                  else:
                        print("pre_or_post must be 'pre' or 'visso' or 'post'")
                  label.at[int(i), "label"] = lab
            frames[f] = frames[f].assign(label=label)
    else:
        print("pre_or_post must be 'pre' or 'visso' or 'post'")
    return frames

    

def add_TTF_in_sec(row):
  """It takes a row from a df and it computes the difference in seconds between the event in the input row and the main. 
  This is called Time To Failure (TTF)
  Args:
  ----------
  row : pandas.core.series.Series
        row from Input DataFrame where to add column TTF
  Returns:
  ----------
  difference : float of the amount of time in seconds between the event in the input row and the main one
        
  """
  time=row['source_origin_time']
  norcia_datetime= datetime.strptime('2016-10-30T07:40:17.000000Z', '%Y-%m-%dT%H:%M:%S.%fZ')
  difference = (time-norcia_datetime).total_seconds()
  return difference

def train_val_test_split(df, train_percentage=0.70, val_percentage=0.10, test_percentage=0.20, force_in_test=[], split_random=True, plot_hist=True):
  """It takes the given df and it splits it in train,val,test and each one is further splitted in X (features: channels), y (label: [1,0]=pre or [0,1]=post), index to
  easily find the sample in the given df).

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        Input DataFrame to be splitted
  train_percentage : float 0<=train_percentage<=1 (Default=0.7)
        Percentage of df length to use as training data.
  val_percentage : float 0<=val_percentage<=1 (Default=0.1)
        Percentage of df length to use as validation data.
  test_percentage : float 0<=test_percentage<=1 (Default=0.2)
        Percentage of df length to use as testing data.
        Note: train_percentage+val_percentage+test_percentage must be <1 and should be =1.
  force_in_test : list
        traces to be forced in test set
  plot_hist : Bool (Default=True)
        If true it plots the hist for classes
  split_random : Bool (Default=True)
        If true it splits the dataset randomly, otherwise it divides the classes setting val and test in the middle of the class, based on trace_start_time 

  Returns:
  ----------
  numpy.ndarray
        split in train,val,test and each one is further splitted in X (features: channels), y (label: [1,0]=pre or [0,1]=post), index to
        easily find the sample in the given df). 
        
  """
  seed_everything(42)
  if (train_percentage+val_percentage+test_percentage)>1:
      print("WARNING: train_percentage+val_percentage+test_percentage cannot be grater than 1")
  
  if split_random:
      # this is to avoid having the same event in train and val/test.
      # If the dataset comes from a single station, this does nothing but shuffling data.
      # old version source_id_array=df.groupby(['source_id']).unique() # we can do the same with df['source_id'].sum()
      # source_id_array=source_id_array.index.to_numpy() 

      source_id_array=list(df.groupby(['source_id']).groups.keys())
      source_id_array=np.array(source_id_array) 

      #np.random.seed(123) 
      source_id_test=[]
      for i in force_in_test:
            if i not in df['trace_name'].values:
                  print("WARNING: ", i," not in df. This will cause an error.")
            source_id_test.append(df.loc[df['trace_name'] == i]['source_id'].values[0])
      source_id_test=np.array(source_id_test)
      source_id_array = np.setdiff1d(source_id_array,source_id_test) # remove indexes that contains traces for testing
      np.random.shuffle(source_id_array)
      source_id_array = np.append(source_id_array, source_id_test, axis=0) # add indexes that contains traces for testing, so that we ensure they ends up in test set
      
      source_id_train=source_id_array[:int(len(source_id_array)*train_percentage)]
      source_id_val=source_id_array[int(len(source_id_array)*train_percentage):int(len(source_id_array)*(train_percentage+val_percentage))]
      source_id_test=source_id_array[int(len(source_id_array)*(train_percentage+val_percentage)):]
  else:
      # in this way we add a column with the corresponding label
      label_df = pd.DataFrame(columns=['label_index','trace_name'])
      for index, row in df.iterrows():
            label_df.at[index, "label_index"] = row['label'].index(max(row['label'])) # index of the class
            label_df.at[index, "trace_name"] = row['trace_name']
      df = pd.merge(df,label_df, on=['trace_name'])

      frames_class = [ df[df['label_index'] ==i] for i in range(len(df.iloc[0]['label']))]
   
      # this is to avoid having the same event in train and val/test.
      # If the dataset comes from a single station, this does nothing but shuffling data.
      source_id_train=[]
      source_id_val=[]
      source_id_test=[]
      for df_frame in frames_class:
            df_frame=df_frame.sort_values(by=['trace_start_time'])
            source_id_array=df_frame['source_id'].unique() # we can do the same with df.groupby(['source_id']).sum() and then source_id_array=source_id_array.index.to_numpy() 
            
            source_id_test_forced=[]
            for i in force_in_test:
                  if i not in df_frame['trace_name'].values:
                        print("WARNING: ", i," not in df_frame. This will cause an error.")
                  source_id_test_forced.append(df_frame.loc[df_frame['trace_name'] == i]['source_id'].values[0])
            source_id_test_forced=np.array(source_id_test_forced)
            source_id_array = [i for i in source_id_array if i not in source_id_test_forced] # remove indexes that contains traces for testing. The following does the same, but sorting elements: 
                                                                                                # source_id_array = np.setdiff1d(source_id_array,source_id_test_forced)
            n_traces_train=int(len(source_id_array)*(train_percentage))
            n_traces_val=int(len(source_id_array)*(val_percentage))
            n_traces_test=int(len(source_id_array)*(test_percentage))
            
            source_id_train_start=source_id_array[:int(n_traces_train/2)]
            source_id_train_end=source_id_array[len(source_id_array)-int(n_traces_train/2):]
            source_id_train_frame = np.append(source_id_train_start, source_id_train_end, axis=0)
            source_id_val_frame=source_id_array[int(n_traces_train/2):int(n_traces_train/2)+n_traces_val]
            source_id_test_frame=source_id_array[int(n_traces_train/2)+n_traces_val:int(n_traces_train/2)+n_traces_val+n_traces_test]
            source_id_test_frame = np.append(source_id_test_frame, source_id_test_forced, axis=0)

            source_id_train = np.append(source_id_train, source_id_train_frame, axis=0).astype(int)
            source_id_val = np.append(source_id_val, source_id_val_frame, axis=0).astype(int)
            source_id_test = np.append(source_id_test, source_id_test_frame, axis=0).astype(int)  
  
  print("Events in train dataset: ",len(source_id_train))
  print("Events in validation dataset: ",len(source_id_val))
  print("Events in test dataset: ",len(source_id_test))
  train_df=df.loc[df['source_id'].isin(source_id_train)]
  train_df = train_df.sample(frac=1).reset_index(drop=True)
  val_df=df.loc[df['source_id'].isin(source_id_val)]
  val_df = val_df.sample(frac=1).reset_index(drop=True)
  test_df=df.loc[df['source_id'].isin(source_id_test)]
  test_df = test_df.sample(frac=1).reset_index()
  df=pd.concat([train_df, val_df])
  df=pd.concat([df, test_df])
  df=df.reset_index(drop=True)

  train_size = len(train_df) #int(len(df) * train_percentage)
  val_size = len(val_df) #int(len(df) * val_percentage)
  test_size = len(test_df) #int(len(df) * test_percentage)

  df_E_channel_norm=pd.DataFrame(df["E_channel"].to_list())
  dataset_trainE = df_E_channel_norm[0:train_size]
  training_setE = dataset_trainE.iloc[:, 0:dataset_trainE.shape[1]].to_numpy(na_value=0)
  training_setE = np.expand_dims(training_setE, axis=2)
  dataset_valE = df_E_channel_norm[train_size:train_size+val_size]
  val_setE = dataset_valE.iloc[:, 0:dataset_valE.shape[1]].to_numpy(na_value=0)
  val_setE = np.expand_dims(val_setE, axis=2)
  dataset_testE = df_E_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setE = dataset_testE.iloc[:, 0:dataset_testE.shape[1]].to_numpy(na_value=0)
  test_setE = np.expand_dims(test_setE, axis=2)

  df_N_channel_norm=pd.DataFrame(df["N_channel"].to_list())
  dataset_trainN = df_N_channel_norm[0:train_size]
  training_setN = dataset_trainN.iloc[:, 0:dataset_trainN.shape[1]].to_numpy(na_value=0)
  training_setN = np.expand_dims(training_setN, axis=2)
  dataset_valN = df_N_channel_norm[train_size:train_size+val_size]
  val_setN = dataset_valN.iloc[:, 0:dataset_valN.shape[1]].to_numpy(na_value=0)
  val_setN = np.expand_dims(val_setN, axis=2)
  dataset_testN = df_N_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setN = dataset_testN.iloc[:, 0:dataset_testN.shape[1]].to_numpy(na_value=0)
  test_setN = np.expand_dims(test_setN, axis=2)

  df_Z_channel_norm=pd.DataFrame(df["Z_channel"].to_list())
  dataset_trainZ = df_Z_channel_norm[0:train_size]
  training_setZ = dataset_trainZ.iloc[:, 0:dataset_trainZ.shape[1]].to_numpy(na_value=0)
  training_setZ = np.expand_dims(training_setZ, axis=2)
  dataset_valZ = df_Z_channel_norm[train_size:train_size+val_size]
  val_setZ = dataset_valZ.iloc[:, 0:dataset_valZ.shape[1]].to_numpy(na_value=0)
  val_setZ = np.expand_dims(val_setZ, axis=2)
  dataset_testZ = df_Z_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setZ = dataset_testZ.iloc[:, 0:dataset_testZ.shape[1]].to_numpy(na_value=0)
  test_setZ = np.expand_dims(test_setZ, axis=2)

  df_label=pd.DataFrame(df["label"].to_list())
  dataset_trainlabel = df_label[0:train_size]
  y_train = dataset_trainlabel.iloc[:, 0:dataset_trainlabel.shape[1]].to_numpy(na_value=0)
  dataset_vallabel = df_label[train_size:train_size+val_size]
  y_val = dataset_vallabel.iloc[:, 0:dataset_vallabel.shape[1]].to_numpy(na_value=0)
  dataset_testlabel = df_label[train_size+val_size:train_size+val_size+test_size]
  y_test = dataset_testlabel.iloc[:, 0:dataset_testlabel.shape[1]].to_numpy(na_value=0)

  df_index=pd.DataFrame(df.index.to_list())
  dataset_trainindex = df_index[0:train_size]
  index_train = dataset_trainindex.iloc[:, 0:dataset_trainindex.shape[1]].to_numpy(na_value=0)
  dataset_valindex = df_index[train_size:train_size+val_size]
  index_val = dataset_valindex.iloc[:, 0:dataset_valindex.shape[1]].to_numpy(na_value=0)
  dataset_testindex = df_index[train_size+val_size:train_size+val_size+test_size]
  index_test = dataset_testindex.iloc[:, 0:dataset_testindex.shape[1]].to_numpy(na_value=0)

  X_train=np.append(training_setE,training_setN, axis=2)
  X_train=np.append(X_train, training_setZ, axis=2)
  X_val=np.append(val_setE,val_setN, axis=2)
  X_val=np.append(X_val, val_setZ, axis=2)
  X_test=np.append(test_setE,test_setN, axis=2)
  X_test=np.append(X_test, test_setZ, axis=2)
  return df, X_train, y_train, index_train, X_val, y_val, index_val, X_test, y_test, index_test

def create_dataloader(X, y, index, target_dataset, batch_size = 32):
  """It takes the given numpy.ndarrays and it changes torch.utils.data.DataLoader, making data suitable for the model training.

  Args:
  ----------
  X : numpy.ndarray
        Model features: channels
  y : numpy.ndarray
        Model label: [post,pre]
  index : numpy.ndarray
        indexes: to easily find the sample in the given df
  batch_size : int (Default=32)
        Size of the batch used during the training of the model.
  target_dataset: string
        "train_dataset" or "val_dataset" or "test_dataset". This choice is to correctly select "shuffle"
        and "drop_last" parameters in torch.utils.data.DataLoader function

  Returns:
  ----------
  dl : torch.utils.data.DataLoader made of X,y,index

  """
  src, lab, idx = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(index)
  src=torch.nn.functional.normalize(src) # <- it normalizes using torch function
  
  dataset = TensorDataset(src, lab, idx)
  if target_dataset=="train_dataset":
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  elif target_dataset=="val_dataset":
    dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  elif target_dataset=="test_dataset":
    dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
  else:
    print("target_dataset not valid.")
  return dl


