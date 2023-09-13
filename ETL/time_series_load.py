import pandas as pd
import numpy as np
import copy
import os

def ensure_dir_exists(directory):
    if os.path.exists(directory) == False:
        os.mkdir(directory)
        print("Creating Directory: ", directory)
    else:
        print("Directory Exists: ",  directory) 
        pass

def set_relative_index(data):
    'Return a copy of a pandas Series or Dataframe with relative index.'
    data = data.copy()
    try:
        data.index -= data.index[0]
        return data
    except Exception as e:
        print(data)

def gather(
    data_dir_path,
    tags_config,
    save_path = '',
    batch_ids = None):

    tags = {}
    for sensor in tags_config:
        tags[sensor] = pd.DataFrame()

    for subdir, dirs, files in os.walk(data_dir_path):
        for file in files:
            sensor = os.path.basename(subdir)
            batch = os.path.basename(os.path.dirname(subdir))

            load_batch = True
            if batch_ids is not None:
                if batch not in batch_ids:
                    load_batch = False
            
            if load_batch:
                csv_file = os.path.join(subdir, file)
                try:
                    data = pd.read_csv(csv_file).drop(['entity'],axis=1)
                    data['batch'] = batch
                    tags[sensor] = tags[sensor].append(data)
                except:
                    pass

    if save_path == '':
        save_path == data_dir_path
    else:  
        ensure_dir_exists(save_path)
    
    phase = os.path.basename(os.path.normpath(data_dir_path))
    for k in tags.keys():
        tags[k].to_csv(os.path.join(save_path,f'{phase}_{k}.csv'), index = None)


def load(
        tag_data, 
        batch_list = None):
    
    tag_data['t'] = pd.to_datetime(tag_data['t'])
    tag_data = tag_data.sort_values('t')
    tag_data = tag_data[tag_data.batch.isin(batch_list)]
    list_of_series = [
            pd.Series(v['v'].values, index=v['t'], name=k)
            for k, v in tag_data.groupby('batch')['t', 'v']
        ]
    
    print(f'Loaded {len(list_of_series)} batches.')
    print(f'Min Timestamp = {tag_data.t.min()}')
    print(f'Max Timestamp = {tag_data.t.max()}')
    print()

    return list_of_series


def resample(
        tag_data, 
        n_chunks = 12, 
        freq = None, 
        name = ''):

    if isinstance(tag_data, list):
        list_of_series = tag_data
    else:
        list_of_series = load(tag_data)

    if freq is None:
        list_of_series_cropped = [s[:-np.mod(s.shape[0],n_chunks)] if np.mod(s.shape[0],n_chunks) != 0 else s for s in list_of_series]
        list_of_series_cropped = [i for i in list_of_series_cropped if len(i) != 0]
        
        subsets_d = [np.array_split(set_relative_index(d), n_chunks) for d in list_of_series_cropped]
        index = [d.name for d in list_of_series_cropped]
        column_names = [f'{str(i)}_{name}' for i in range(1, n_chunks+1)]
            
        s_df = []
        for sub in subsets_d:
            mean_s_d = [s.mean() for s in sub]
            s_df.append(mean_s_d)

        df_res = pd.DataFrame(s_df, columns=column_names).set_index(pd.Index(index), 'batch')
    else:
        df_res = pd.concat([set_relative_index(d).resample(freq).mean() for d in list_of_series], axis=1)

    if all(df_res.iloc[:, 0] == 0):
        df_res = df_res.drop(df_res.columns[0], axis=1)

    return df_res


def derivative(
        data, 
        how = 'pct', 
        group_by = None):
    
    if group_by is None:
        if how == 'diff':
            df_res = data.apply(lambda x: x.diff(), axis=1)
        elif how == 'pct':
            df_res = data.apply(lambda x: x.pct_change(), axis=1)
        else:
            pass
        df_res = df_res.iloc[: , 1:]
        df_res.columns = ['diff_' + c for c in df_res.columns.tolist()]
        return df_res
    else:
        pass

def load_and_transform(
                    tags_dict,  
                    batch_list):

    tags_dict_filled = {}
    
    for tag in tags_dict:
            print(f'Loading: {tag}...')

            info = tags_dict[tag]
            file = info['file']
            res = info['resample']
            deriv = info['derivative']
            data = pd.read_csv(file)

            resampling = True
            if type(res) == int:
                freq = None
                n_chunks = res
            elif type(res) == str:
                freq = res
            else:
                resampling = False
        
            difference = True
            if deriv == 'pct':
                how = 'pct'
            elif deriv == 'diff':
                how = 'diff'
            else:
                difference = False

            loaded_data = load(data, batch_list)
            if resampling:
                data_res = resample(loaded_data, n_chunks = n_chunks, freq = freq, name = tag)
                tags_dict_filled[tag] = data_res
                
                if difference:
                    tags_dict_filled['diff_' + tag] = derivative(data_res, how = how)
                else:
                    pass
    
    return tags_dict_filled
        
    

