import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def plot_clustermap_with_left_bar(
    data,
    bar_data=None,
    build_palette=False,
    show_colorbar=True,
    method='average',
    metric='euclidean',
    pdf_obj=None,
):

    if build_palette and bar_data is not None:
        upper_bar = bar_data.astype('int')
        pa = sns.color_palette(palette='Set2', n_colors=upper_bar.max().max() + 1)
        upper_bar = upper_bar.applymap(lambda x: pa[x])
    else:
        upper_bar = bar_data

    d = data.loc[:, (data != 0).any(axis=0)]

    cm = sns.clustermap(
        d,
        figsize=(12, 10),
        method='average',
        metric='cosine',
        vmax=0.4,
        cbar_kws={'extend': 'max', 'extendfrac': 0.2},
        dendrogram_ratio=0.13,
        cbar_pos=(0.05, 0.86, 0.05, 0.12),
        rasterized=True,
    )

    # cm = sns.clustermap(data, method=method, metric = metric, row_colors = upper_bar, rasterized = True)

    if not show_colorbar:
        cm.cax.set_visible(False)

    if pdf_obj != None:
        pdf_obj.savefig()

    return cm


def plot_heatmap_with_left_bar(
    data,
    var,
    sort_batch_by,
    left_bar_CQA,
    normalize=True,
    pdf_obj=None,
    l_color_norm_by_cluster=False,
    vmax=None,
    vmin=None,
    title='',
    outer_spec=None,
    fig=None,
    xtickrotation=90
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
    data = data.sort_values(sort_batch_by)
    l_bar = data[left_bar_CQA]
    l_bar = (l_bar - l_bar.mean()) / l_bar.std()  # if isinstance(left_bar_CQA, list) else l_bar

    if outer_spec != None:
        inner_spec = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_spec, width_ratios=[1, 5]
        )
        a0 = plt.Subplot(fig, inner_spec[0])
        a1 = plt.Subplot(fig, inner_spec[1])
    else:
        f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]}, figsize=(20, 10))
        f.suptitle(title)

    g_l = sns.heatmap(l_bar, ax=a0, rasterized=True, cmap='viridis', yticklabels=False, robust=True)
    g_l.set_xticklabels(g_l.get_xticklabels(), rotation=xtickrotation, ha='right')

    d = (data[var] - data[var].mean()) / data[var].std() if normalize else data[var]
    g = sns.heatmap(
        d, ax=a1, rasterized=True, robust=True, vmax=vmax, vmin=vmin, cmap='Reds', yticklabels=False
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=xtickrotation, ha='right')

    if outer_spec != None:
        fig.add_subplot(a0)
        fig.add_subplot(a1)
        a1.set_title(title)

    if pdf_obj != None:
        pdf_obj.savefig()

    return (a0, a1)
