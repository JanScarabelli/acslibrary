import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from string import Template
import numpy as np

def plot_profile(
    tag_data,
    agg_data,
    cat_var,
    figsize=(20, 20),
    ylabel=None,
    pdf=None,
    max_length=None,
    alpha=0.01,
    ylim=None,
    return_mean_profile=None,
):
    """
    plot the profile of time series separated by a categorical variable (e.g. cluster)

    tag_data: timeserie data. index should be the batch ID (same as agg_data)
    agg_data: should contain the cat_var column, and the index should be the batch ID
    cat_var: categorical variable
    figsize: tuple of integers
    pdf: pdf object or None
    """

    cats = list(agg_data[cat_var].unique())
    cats.sort()
    fig, ax = plt.subplots(len(cats) + 1, 1, figsize=figsize, sharex=True, sharey=True)

    common_batches = list(set(agg_data.index) & set(tag_data.index))
    agg_data = agg_data.loc[common_batches]
    tag_data = tag_data.loc[common_batches]

    if max_length is not None:
        tag_data = tag_data.loc[:, tag_data.columns < max_length]

    for k in range(len(cats)):
        for b in agg_data.loc[agg_data[cat_var] == cats[k]].index:
            try:
                ax[k].plot(
                    tag_data.columns, tag_data.loc[b], alpha=alpha, c='k'
                )
            except Exception as e:
                print(e)
        ax[k].set_xlabel('hours')
        ax[k].set_ylabel(ylabel) if ylabel is not None else None
        ax[k].set_title('category: ' + str(cats[k]))
        ax[k].set_ylim(ylim) if ylim is not None else None
        mean_df = tag_data.loc[agg_data.loc[agg_data.loc[common_batches,cat_var]==k].index].mean(0)

        ax[len(cats)].plot(tag_data.columns, mean_df, alpha=1, label=k)
        ax[len(cats)].legend()
        ax[len(cats)].set_xlabel('hours')
        ax[len(cats)].set_ylabel(ylabel) if ylabel is not None else None
        ax[len(cats)].set_title('Average profile for all categories')
        ax[len(cats)].set_ylim(ylim) if ylim is not None else None
    pdf.savefig() if pdf is not None else plt.show()
    plt.close('all')
    if return_mean_profile is not None:
        ret = pd.DataFrame()
        for k in return_mean_profile:
            df = pd.DataFrame()
            mean = tag_data.loc[agg_data.loc[agg_data.loc[common_batches,cat_var]==k].index].mean(0)
            df[['time','mean_profile']] = mean.reset_index()
            df[cat_var] = k
            ret = pd.concat([ret, df], axis=0)
        return ret


def plot_scada_heatmap_with_left_bar(
    tag_data,
    agg_data,
    sort_batch_by,
    left_bar_CQA,
    sort_var,
    title = '',
    pdf=False,
    vmax=None,
    vmin=None,
    val_cnt=False,
    max_length=None,
    start=None,
    end=None,
):
    """
    plots a heatmap of all timeseries belonging to a same tag and phase, sorted by the values of
    the argument 'to_be_sorted_by'.
    Arguments:
    ------------
    to_be_sorted_by: string
        the agg_data column to use to sort the bacthID. You should use a CQA
    bycluster: int {0, 1, 2, 3}
        default is None (the heatmap will have all the batches). If you indicate one of the numbers,
        we filter the batches
    """

    if max_length is not None:
        tag_data = tag_data.loc[:, tag_data.columns < max_length]

    df = tag_data.loc[
        agg_data[agg_data.index.isin(tag_data.index)].sort_values(sort_batch_by).index
    ]
    try:
        f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]}, figsize=(30, 15))
        try:
            title = 'Ordered by ' + str(sort_batch_by) + ' - ' + title
            if (start != None) & (end != None):
                title += ' - timeranges: ' + f'[{start}, {end}]'
            plt.title(title)
        except Exception as e:
            print(e, 'when trying to set the title')
        l_bar = pd.DataFrame(agg_data.loc[df.index][left_bar_CQA])
        if val_cnt != False:
            print(l_bar[val_cnt].value_counts())
        l_bar = ((l_bar - l_bar.mean()) / l_bar.std()) if isinstance(left_bar_CQA, list) else l_bar
        sns.heatmap(l_bar, ax=a0, rasterized=True, cmap='viridis', yticklabels=False, robust=True)
        batch_dates = agg_data.loc[df.index][sort_var]
        yticks_dates = (
            'auto' if ((sort_batch_by == sort_var) | (sort_batch_by == [sort_var])) else False
        )
        try:
            df.columns = df.columns.to_series().apply(lambda x: strfdelta(x, '%H:%M:%S'))
            df.index = batch_dates.dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(e, 'this one we do not have time columns anymore')
        sns.heatmap(
            df.astype('float64'),
            ax=a1,
            rasterized=True,
            robust=True,
            yticklabels=yticks_dates,
            vmax=vmax,
            vmin=vmin,
            cmap='Reds',
        )
        if pdf:
            pdf.savefig()
        else:
            plt.show()
        plt.close('all')
    except Exception as ex:
        plt.close('all')
        print(ex)
        pass


def strfdelta(tdelta, fmt):
    class DeltaTemplate(Template):
        delimiter = '%'

    d = {}
    total_seconds = int(tdelta.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d['H'] = f'{hours:02d}'
    d['M'] = f'{minutes:02d}'
    d['S'] = f'{seconds:02d}'
    t = DeltaTemplate(fmt)
    return t.substitute(**d)
