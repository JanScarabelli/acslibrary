import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def waterfall_plot(data, sort_var, var, freq='M', pdf_obj=None):
    """
    Outputs a waterfall plot
    data: dataframe
    sort_var: the datetime variable that we'll use for frequency analysis
    var: the var that we're interested in the frequency
    count_var: an additional variable so that we can count points, must be different than var
    freq: 'M', 'W', 'D', ... frequency to apply to the Group
    """
    # clusters = list(data[var].dropna().unique()) if drop_na_cat else list(data[var].unique())
    # clusters.sort()
    # data[sort_var] = pd.to_datetime(data[sort_var])
    # data_grouped = data.groupby([pd.Grouper(key=sort_var,freq=freq),pd.Grouper(var)])[count_var].count().reset_index()

    # idx = pd.MultiIndex.from_product([data_grouped[sort_var].unique(), clusters], names=[sort_var, var])
    # data_grouped = data_grouped.set_index([sort_var, var]).reindex(idx).reset_index().fillna(0)

    # data_grouped = data_grouped.sort_values([sort_var, var])
    # data_grouped['perc'] = data_grouped.groupby(sort_var)[count_var].apply(lambda x: round((x/x.sum())*100,2))

    # var_dicts = data_grouped.groupby(var)['perc'].apply(list).to_dict()

    # x = data_grouped[sort_var].unique()
    # y = np.array(list(var_dicts.values()))

    # plt.figure(figsize=(15,10))
    # col = sns.color_palette("Paired", 12)
    # plt.stackplot(x,y, labels=clusters, colors=col)
    # plt.legend(loc='upper left')
    wf = data.copy()
    wf[var] = wf[var].astype('category')
    wf = wf[[var, sort_var]].set_index(sort_var).groupby([pd.Grouper(freq=freq)]).value_counts()
    wf = wf.reset_index().rename(columns={0: 'clus_count'})
    wf['perc'] = wf.groupby(sort_var, group_keys=False)['clus_count'].apply(lambda x: round((x / x.sum()) * 100, 2)) # changed

    var_dicts = wf.groupby(var)['perc'].apply(list).to_dict()
    x = wf[sort_var].unique()
    y = np.array(list(var_dicts.values()))

    plt.figure(figsize=(15, 10))
    col = sns.color_palette('Paired', 12)
    plt.stackplot(x, y, labels=var_dicts.keys(), colors=col)
    plt.legend(loc='upper left')

    if pdf_obj == None:
        plt.show()
    else:
        pdf_obj.savefig()
