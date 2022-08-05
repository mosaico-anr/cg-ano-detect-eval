import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader



network_conditions = ['excellent', 'very_good', 'good', 'average', 'bad', 'highway']
cg_platforms = ['std', 'xc', 'gfn', 'psn']

#col_drop = ['count', 'first_frame_received_to_decoded_ms', 'frames_rendered', 'interframe_delay_max_ms', 'current_delay','target_delay_ms', 'decode_delay',
#           'jb_cuml_delay', 'jb_emit_count', 'sum_squared_frame_durations', 'sync_offset_ms', 'total_bps', 'total_decode_time_ms',  'total_frames_duration_ms',
#           'total_freezes_duration_ms', 'total_inter_frame_delay', 'total_pauses_duration_ms', 'total_squared_inter_frame_delay', 
#           'max_decode_ms', 'render_delay_ms', 'min_playout_delay_ms', 'dec_fps', 'ren_fps', 'cdf', 'packetsReceived', 'packetsLost', 'time_ms']
to_keep = ['time_ms', 'decode_delay', 'jitter','jb_delay', 'packetsReceived_count',
            'net_fps', 'height', 'width', 'frame_drop','frames_decoded', 'rtx_bps', 'rx_bps', 'freeze',
            'throughput','rtts']

def read_csv_files(path):
    """Read and load the csv files.

    Args:
        path (str): The path of a csv file data.

    Returns:
            The pandas Dataframe.
    """
    time_step = 5
    df = pd.read_csv(path)
    df['packetsReceived_count'] = [0.0] + [curr - previous for previous,curr in zip(df['packetsReceived'].values, df['packetsReceived'].iloc[1:].values)]
    df['time_ms'] = pd.to_timedelta(df['time_ms'], unit='ms')
    df = df[to_keep]
    df = df.dropna(axis=0)
    df = df.set_index('time_ms').resample(f"{time_step}ms").last()
    df = df.reset_index()
    return df.dropna(axis=0)


def get_data_dict(data_path):
    """Load and merge the data from CG outputs.

    Args:
        data_path (str): The data folder path.

    Returns:
            The dict with all the data.
    """
    csv_path_1 = os.path.join(data_path,'racing_std_1/')
    csv_path_2 = os.path.join(data_path, 'racing_std_2/')
    df_dicts = {}
    
    for ntw_cnd in network_conditions:
        path_1 = f"{csv_path_1+ntw_cnd}.csv"
        path_2 = f"{csv_path_2+ntw_cnd}.csv"
    
        df1 = read_csv_files(path_1)
        df2 = read_csv_files(path_2)
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df_dicts[ntw_cnd] = df
    return df_dicts

def label_on_conditions(df):
    """The labels creation of the datasets according to CG platforms recommendations.

    Args:
        df (pd.Dataframe): The raw dataframe.
    """
    conditions = [
        (df['height'] == 720.0) | (df['freeze'] == 1.0) | (df['net_fps'] < 60.0),
        (df['height'] == 1080.0) & (df['freeze'] == 0.0) & (df['net_fps'] >= 60.0)
        ]
    values = [1.0, 0.0]
    df['anomaly'] = np.select(conditions, values)
    
def get_sub_seqs(x_arr, window_size, stride=1, start_discont=np.array([])):
    """Process the data into moving window.

    Args:
        x_arr (np.array)                    : The data arrays. 
        window_size (int)                   : The window size.
        stride (int, optional)              : The stride value. Defaults to 1.
        start_discont (np.array, optional)  : Defaults to np.array([]).

    Returns:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - window_size + 1), start)) for start in start_discont if start > window_size]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - window_size + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + window_size] for i in seq_starts])
    return x_seqs

def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits,
                          seed: int, shuffle: bool = False, usetorch = True):
    """[summary]

    Args:
        x_seqs (np.ndarray): [description]
        batch_size (int): [description]
        splits ([type]): [description]
        seed (int): [description]
        shuffle (bool, optional): [description]. Defaults to False.
        usetorch (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if np.sum(splits) != 1:
        scale_factor = np.sum(splits)
        splits = [fraction/scale_factor for fraction in splits]
    if shuffle:
        np.random.seed(seed)
        x_seqs = x_seqs[np.random.permutation(len(x_seqs))]
        np.random.seed()
    split_points = [0]
    for i in range(len(splits)-1):
        split_points.append(split_points[-1] + int(splits[i]*len(x_seqs)))
    split_points.append(len(x_seqs))
    if usetorch:
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]], batch_size=batch_size,
            drop_last=False, pin_memory=True, shuffle=False) for i in range(len(splits))])
        return loaders
    else:
        # datasets = tuple([x_seqs[split_points[i]: 
        #     (split_points[i] + (split_points[i+1]-split_points[i])//batch_size*batch_size)] 
        #     for i in range(len(splits))])
        datasets = tuple([x_seqs[split_points[i]:split_points[i+1]]
            for i in range(len(splits))])
        return datasets
    

    
def train_test_split(normal_df, anomaly_df, seed=42):
    """Split the data into train-test set.

    Args:
        normal_df (pd.Dataframe)        : The dataframe of normal observations.
        anomaly_df (pd.Dataframe)       : The dataframe of anomalous observations.
        seed (int, optional)            : The random seed generator. Defaults to 42.

    Returns:
    """
    # Split normal df in 50 - 50 randomly
    shuffled_indices = np.random.RandomState(seed=seed).permutation(len(normal_df))
    test_ratio = 0.5
    test_set_size = int(len(normal_df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_normal, test_normal = normal_df.iloc[train_indices], normal_df.iloc[test_indices]
    
    # Split anomaly test df and contamination set
    shuffled_indices = np.random.RandomState(seed=seed).permutation(len(anomaly_df))
    contamination_ratio = 0.4
    contamination_set_size = int(len(anomaly_df) * contamination_ratio)
    contamination_set_indices = shuffled_indices[:contamination_set_size]
    test_anomaly_indices = shuffled_indices[contamination_set_size:]
    contamination_df, test_anomaly = anomaly_df.iloc[contamination_set_indices], anomaly_df.iloc[test_anomaly_indices]
    
    
    train = pd.concat([train_normal, contamination_df], axis=0, ignore_index=True)
    train = train.sample(frac=1, ignore_index=True, random_state=seed)
    
    test = pd.concat([test_normal, test_anomaly], axis=0, ignore_index=True)
    test = test.sample(frac=1, ignore_index=True, random_state=seed)
    
    return train, test


def split_contamination_data(df, contamination_ratio, seed=42):
    """The creation of the training data with contamination_ratio % of anomalous samples.

    Args:
        df (pd.Dataframe)           : The training dataframe. 
        contamination_ratio (float) : The contamination ratio, between 0 and 1.
        seed (int, optional)        : The random generator seed. Defaults to 42.

    Raises:
        ValueError                  : If the contamination ratio is not between 0 and 1.

    Returns:
            The training data with contaminated data.
    """
    anomaly_index_list = df[df['anomaly'] == 1.0].index.values
    #print(len(anomaly_index_list))
    
    if contamination_ratio==0:
        return df.drop(index=anomaly_index_list)
    elif contamination_ratio==1:
        return df
    
    elif contamination_ratio < 1:
        
        shuffled_indices = np.random.RandomState(seed=seed).permutation(len(anomaly_index_list))
        contamination_set_size = int(len(anomaly_index_list) * contamination_ratio)
        


        # Get the indices of the anomaly to keep
        #anomaly_indices_to_keep = anomaly_index_list[shuffled_indices[:contamination_set_size]]
        anomaly_indices_to_drop = anomaly_index_list[shuffled_indices[contamination_set_size:]]
        
        return df.drop(index=anomaly_indices_to_drop)
        

    else:
        raise ValueError('Contamination ratio must be between 0 and 1 !')

def get_window_labels(labels, window_size=10):
    """The processing of the labels into window labels.

    Args:
        df (pd.Dataframe)           : The dataframe
        window_size (int, optional) : The window size. Defaults to 10.

    Returns:
        The data, the window labels and the labels.
    """
    windows_labels=[]
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int_(labels[i:i+window_size])))

    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
    return y_test

def labels_processing(df, window_size=10):
    """The processing of datasets into data, labels and window labels.

    Args:
        df (pd.Dataframe)           : The dataframe
        window_size (int, optional) : The window size. Defaults to 10.

    Returns:
        The data, the window labels and the labels.
    """
    label = df['anomaly']
    df_wo_labels = df.drop(labels='anomaly', axis=1)
    
    window_labels = get_window_labels(label, window_size=window_size)
    
    return df_wo_labels, window_labels, label

def data_processing_random(normal_df, anomaly_df, contamination_ratio, window_size=10, seed=42):
    """The "mixed-datasets" splitting strategy.

    Args:
        normal_df (pd.Dataframe)        : The dataframe of normal observationsconcatenated.
        anomaly_df (pd.Dataframe)       : The dataframe of anomalous observations concatenated.
        contamination_ratio (float)     : The contamination ratio. Must be between 0 and 1. 
        window_size (int, optional)     : The window size. Defaults to 10.
        seed (int, optional)            : The random generator seed. Defaults to 42.

    Returns:
            The data dict with the train/test and the labels.
    """
    train, test = train_test_split(normal_df, anomaly_df, seed=seed)
    
    df_contamination = split_contamination_data(train, contamination_ratio=contamination_ratio, seed=seed)
    train_data, train_win_labels, train_labels = labels_processing(df_contamination, window_size=window_size)
    
    test_data, test_win_labels, test_labels = labels_processing(test, window_size=window_size)
    
    d = {}
    d['train'] = {'data' : train_data, 'window_labels': train_win_labels, 'labels': train_labels}
    d['test'] = {'data' : test_data, 'window_labels': test_win_labels, 'labels': test_labels}
    return d

def data_processing_naive(train, test, contamination_ratio, window_size=10, seed=42):
    """The "high-bitrate" splitting strategy.

    Args:
        train (pd.Dataframe)            : The training dataframe concatenated 
                                          from the best bitrate CG sesions.
        test (pd.Dataframe)             : The test dataframe concatenated 
                                          from the best bitrate CG sesions.
        contamination_ratio (float)     : The contamination ratio. Must be between 0 and 1. 
        window_size (int, optional)     : The window size. Defaults to 10.
        seed (int, optional)            : The random generator seed. Defaults to 42.

    Returns:
            The data dict with the train/test and the labels.
    """
    
    df_contamination = split_contamination_data(train, contamination_ratio=contamination_ratio, seed=seed)
    train_data, train_win_labels, train_labels = labels_processing(df_contamination, window_size=window_size)
    
    test_data, test_win_labels, test_labels = labels_processing(test, window_size=window_size)
    
    d = {}
    d['train'] = {'data' : train_data, 'window_labels': train_win_labels, 'labels': train_labels}
    d['test'] = {'data' : test_data, 'window_labels': test_win_labels, 'labels': test_labels}
    return d

def get_data(data_path):
    """[summary]

    Args:
        data_path (str): The data csv folder path.

    Returns:
    """
    df_dicts = get_data_dict(data_path)
    for _, df in df_dicts.items():
        label_on_conditions(df)
    
    # Concat the dataframes
    train_cg_1 = df_dicts['excellent']
    train_cg_2 = df_dicts['very_good']
    train_cg_3 = df_dicts['good']
    
    
    train_cg_1 = train_cg_1.drop('time_ms', axis=1)
    train_cg_2 = train_cg_2.drop('time_ms', axis=1)
    train_cg_3 = train_cg_3.drop('time_ms', axis=1)
    
    test_cg_1 = df_dicts['average']
    test_cg_2 = df_dicts['bad']
    test_cg_3 = df_dicts['highway']
    
    test_cg_1 = test_cg_1.drop('time_ms', axis=1)
    test_cg_2 = test_cg_2.drop('time_ms', axis=1)
    test_cg_3 = test_cg_3.drop('time_ms', axis=1)
    
    whole_train = pd.concat([train_cg_1, train_cg_2, train_cg_3], axis=0, ignore_index=True)

    test = pd.concat([test_cg_1,test_cg_2,test_cg_3], axis=0, ignore_index=True)
    test = test.dropna(axis=0)
    
    all_data = pd.concat([whole_train, test], axis=0, ignore_index=True)
    normal_df = all_data[all_data['anomaly']==0.0]
    anomaly_df = all_data[all_data['anomaly']==1.0]
    
    return whole_train, test, normal_df, anomaly_df
    