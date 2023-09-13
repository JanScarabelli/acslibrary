import pandas as pd
import numpy as np
import copy

def set_relative_index(data):
    'Return a copy of a pandas Series or Dataframe with relative index.'
    data = data.copy()
    try:
        data.index -= data.index[0]
        return data
    except Exception as e:
        print("Can't do operations between tz-aware and tz-naive index", e)
        print(data)

def load_data(
    tag, 
    agg_data = None, 
    DL = None, 
    step = None, 
    role = None, 
    equipment = None,
    start=None, 
    end=None, 
    max_length=None, 
    derivative = None, #nobs = None, round_int = None, 
    transposed = False, 
    resample = None,
    tz=None, 
    raw = False, 
    name = 'value', 
    chunks = None):
    """
    step: str or list or None
    role: str or list or None
    equipment: str or list or None
    tag: str or list or None
    start: str or None
    end: str or None
    tz: only if start/end are filled, create a tz-aware timestamp for start/end.
    tz example: 'Europe/Madrid' or 'America/Los_Angeles'
    tz note: if everything is already in UTC, then set to None

    Note:
    DL is an instance of data_loading_tools.DataLoader. It needs to be set up before.
        from data_loading_tools import DataLoader
        DL = DataLoader('../data/prepared_time_series/')
    """

    from_file = False
    if DL is None:
        from_file = True

    if from_file:
        tag_data = pd.read_csv(tag)
        #tag_data['t'] = tag_data.t.map(lambda x: pd.Timestamp(x).tz_localize(tz, ambiguous=False))
        tag_data['t'] = pd.to_datetime(tag_data['t'])
        tag_data = tag_data.sort_values('t')
        tag_data = [
            pd.Series(v['v'].values, index=v['t'], name=k)
            for k, v in tag_data.groupby('batch')['t', 'v']
        ]
    else:
        tag_data = DL.load(batch=None, step=step, role=role, equipment=equipment, tag=tag)

    df_res = '_'

    if (start is not None) | (end is not None):
        start_end_df = agg_data[[start, end]]
    tag_data_reduced = []
    for d in tag_data:

        if from_file:
            id = d.name
        else:
            id = d.name[0]

        if agg_data is not None:
            if id in agg_data.index:
                if (start is not None) | (end is not None):
                    s = pd.Timestamp(start_end_df.loc[id][start], tz=tz)
                    e = pd.Timestamp(start_end_df.loc[id][end], tz=tz)
                    dd = d.loc[(d.index > s) & (d.index < e)]
                else:
                    dd = d.copy()

                if not dd.empty:
                    # dd = pd.DataFrame(dd).reset_index()
                    # dd.columns = ['time','value']
                    # dd['batch'] = d.name[0]
                    tag_data_reduced.append(dd)
                else:
                    pass
                    #print(id + " is empty")
            else:
                pass
                #print(id + ' not in agg data')
        else:
            tag_data_reduced.append(d)

    print(f'{len(tag_data_reduced)} batches were loaded.')

    # df = pd.concat([d for d in tag_data_reduced], axis = 0)
    # df = df.reset_index()
    # return df, '_'

    if resample is not None:
        if chunks:
            tag_data_reduced_cropped = [s[:-np.mod(s.shape[0],chunks)] if np.mod(s.shape[0],chunks) != 0 else s for s in tag_data_reduced]
            tag_data_reduced_cropped = [i for i in tag_data_reduced_cropped if len(i) != 0]
            
            subsets_d = [np.array_split(set_relative_index(d), chunks) for d in tag_data_reduced_cropped]
            print('subsets',len(subsets_d))
            index = [d.name for d in tag_data_reduced_cropped]
            column_names = [f'{str(i)}_{name}' for i in range(1, chunks+1)]
                
            s_df = []
            for sub in subsets_d:
                mean_s_d = [s.mean() for s in sub]
                mean_s_d = [int(m * 1000000) / 1000000.0 for m in mean_s_d]
                s_df.append(mean_s_d)

            df_res = pd.DataFrame(s_df, columns=column_names).set_index(pd.Index(index), 'batch')
        else:
            df_res = pd.concat([set_relative_index(d).resample(resample).mean() for d in tag_data_reduced], axis=1)
    
    if raw:
        df = pd.concat([d for d in tag_data_reduced], axis = 1)
        #df.columns.rename(["batch", "step", "role", "equipment", "tag"], level=[0,1,2,3, 4], inplace=True)
        #df.columns = df.columns.droplevel(['step', 'role', 'equipment', 'tag'])
        df = df.stack().dropna().reset_index()
    else:
        df = pd.concat([set_relative_index(d) for d in tag_data_reduced], axis = 1)
        
    # Cut time series to given length
    if max_length is not None:
            if not raw:
                df = df.loc[df.index<max_length]
            if resample:
                df_res = df_res.loc[df_res.index<max_length]

    if derivative is not None:

        if derivative == 'diff':
            if resample is not None:
                df_res = df_res.apply(lambda x: x.diff(), axis=1)
                df_res = df_res.iloc[: , 1:]
            else:
                df = df.apply(lambda x: x.diff(), axis=1)
                df = df.iloc[: , 1:]

        if derivative == 'pct':
            if resample is not None:
                df_res = df_res.apply(lambda x: x.pct_change(), axis=1)
                df_res = df_res.iloc[: , 1:]
            else:
                df = df.apply(lambda x: x.pct_change(), axis=1)
                df = df.iloc[: , 1:]

    if transposed:  # or expand:
        df_mod = df.copy()

        if not raw and not from_file:
            df_mod.columns.rename(["batch", "step", "role", "equipment", "tag"], level=[0,1,2,3,4], inplace=True)
            df_mod.columns = df_mod.columns.droplevel(['step', 'role', 'equipment', 'tag'])
    
        if not raw:
            df = df_mod.transpose().copy()

            if all(df.iloc[:,0] == 0):
                df = df.drop(df.columns[0],axis=1)

        if resample is not None:

            if not from_file and chunks is None:
                df_res.columns.rename(
                    ['batch', 'step', 'role', 'equipment', 'tag'],
                    level=[0, 1, 2, 3, 4],
                    inplace=True,
                )
                df_res.columns = df_res.columns.droplevel(['step', 'role', 'equipment', 'tag'])

            if chunks is None:
                df_res = df_res.transpose().copy().fillna(0)
                df_res.columns = [f'{name}_{i}' for i in range(0,df_res.shape[1])]
            else:
                df_res = df_res.reset_index().drop(['level_1','level_2','level_3','level_4'],axis=1).set_index('level_0')

            if all(df_res.iloc[:, 0] == 0):
                df_res = df_res.drop(df_res.columns[0], axis=1)

    return df, df_res


def bulk_load_data(
    scada_tags, 
    DL = None, 
    agg_data = None, 
    transposed = True, 
    tz=None, 
    concat_with = None, 
    keep = 'first', 
    raw = False,
    chunks = None):

    scada_tags_filled = copy.deepcopy(scada_tags)

    for t in scada_tags.keys():
        print('loading data...', t)

        if DL is None:
            scada_tags_filled[t]['info']['step'] = None
            scada_tags_filled[t]['info']['equipment'] = None
            scada_tags_filled[t]['info']['role'] = None

        td, td_res = load_data(
                        tag=scada_tags[t]['info']['tag'],
                        step=scada_tags_filled[t]['info']['step'],
                        equipment=scada_tags_filled[t]['info']['equipment'],
                        role=scada_tags_filled[t]['info']['role'],
                        derivative = scada_tags[t]['info']['derivative'],
                        max_length=  scada_tags[t]['graph']['max_length'],
                        start=scada_tags[t]['info']['start'],
                        end=scada_tags[t]['info']['end'],
                        DL=DL,
                        agg_data=agg_data,
                        tz=tz,
                        transposed = transposed,
                        # nobs = nobs,
                        # round_int = round_int,
                        resample = scada_tags[t]['info']['resample'],
                        name = t,
                        raw = raw,
                        chunks=chunks)

        scada_tags_filled[t]['data'] = td
        scada_tags_filled[t]['data_res'] = td_res

        if concat_with is not None:
            if t in concat_with:
                scada_tags_filled[t]['data'] = pd.concat([scada_tags_filled[t]['data'], concat_with[t]['data']])
                scada_tags_filled[t]['data_res'] = pd.concat([scada_tags_filled[t]['data_res'] , concat_with[t]['data_res']])
                dups = scada_tags_filled[t]['data'].index.duplicated(keep = keep)
                scada_tags_filled[t]['data'] = scada_tags_filled[t]['data'][~dups]
                scada_tags_filled[t]['data_res'] = scada_tags_filled[t]['data_res'][~dups]


    return scada_tags_filled
