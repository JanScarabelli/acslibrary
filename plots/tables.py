import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table


def plot_dataframe(
    df, pdf_obj=None, ttle='', col_labels=None, fig_size=(10, 4), outer_spec=None, fig=None
):
    """
    Prints a dataframe
    """
    if outer_spec != None:
        inner_spec = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_spec)
        ax = plt.Subplot(fig, inner_spec[0])
    else:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # if ax is None:
    #    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    ax.title.set_text(ttle)
    ax.axis('tight')
    ax.axis('off')
    table(
        ax, df.round(4), loc='center', rowLabels=None
    )  # , colLabels=[f'{name}_{k}_clusters', 'cluster plasma origin', 'count', 'mean', 'std']

    if outer_spec != None:
        fig.add_subplot(ax)

    if pdf_obj is not None:
        pdf_obj.savefig()

    return ax
