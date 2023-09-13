#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from random import randrange
import random


def _plot_scatter(df, xvar, yvar, alpha=0.4, hue=None, title=None, figsize=(4, 3), outpdf=None):
    f, (ax_1) = plt.subplots(1, figsize=figsize)
    if hue != None:
        sns.scatterplot(x=xvar, y=yvar, data=df, hue=hue, ax=ax_1, alpha=alpha)
    else:
        sns.scatterplot(x=xvar, y=yvar, data=df, ax=ax_1, alpha=alpha)
    if title != None:
        ax_1.set(title=title)
    if outpdf != None:
        outpdf.savefig()
    plt.close()


def all_graphs(
    data,
    target,
    outfile_pdf,
    varlist=[],
    time_vars=None,
    groupvars=None,
    grouppairs=None,
    exclude_vars=None,
    figsize=(9, 5.5),
    allnumvars=False,
    plot1_box_dist_all=True,
    plot2_box_dist_grp=True,
    plot3_box_grp=True,
    plot4_box_pairs_grp=True,
    plot5_reg_all=True,
    plot6_scatter_grp=True,
    plot7_scatter_time_all=True,
    plot8_scatter_time_grp=True,
):
    """
    Parameters
    ----------
    data : dataframe
        Name of dataset.
    target : str
        Name of a single target variable.
    outfile_pdf : str
        Filename with path of output PDF file (ie: '/Users/JoeBlow/test_pdf.pdf').
    varlist : list or str, optional
        List of numeric variables to plot. The 'target' parameter will automatically be included in this list.
        The default is [].
    time_vars : list or str, optional
        List of datetime variables. The default is None.
    groupvars : list, optional
         List of variables to group graphs by. The default is None.
    grouppairs : list, optional
        List of tuples to plot two-way groupings. The default is None.
    exclude_vars : list, optional
        List of variables to exclude from plotting. The default is None.
    figsize : tuple, optional
        Tuple of float values (size of each plot). The default is (9,5.5).
    allnumvars : bool, optional
        Specify True to plot all numeric variables in the dataset (this avoids needing to use the 'varlist' parameter).
        The default is False.
    plot1_box_dist_all : bool, optional
        Choice to generate plot1. The default is True.
    plot2_box_dist_grp : bool, optional
        Choice to generate plot2. The default is True.
    plot3_box_grp : bool, optional
        Choice to generate plot3. The default is True.
    plot4_box_pairs_grp : bool, optional
        Choice to generate plot4. The default is True.
    plot5_reg_all : bool, optional
        Choice to generate plot5. The default is True.
    plot6_scatter_grp : bool, optional
        Choice to generate plot6. The default is True.
    plot7_scatter_time_all : bool, optional
        Choice to generate plot7. The default is True.
    plot8_scatter_time_grp : bool, optional
        Choice to generate plot8. The default is True.

    Returns
    -------
    None.

    """

    with PdfPages(outfile_pdf) as pdf:

        if (time_vars != None) & (type(time_vars) != list):
            time_vars = [time_vars]
        if (groupvars != None) & (type(groupvars) != list):
            groupvars = [groupvars]
        if (varlist != []) & (type(varlist) != list):
            varlist = [varlist]
        if (varlist == []) & (allnumvars):
            varlist = data.select_dtypes(include='number').columns.values

        varlist = [i for i in varlist if i not in [target]]
        varlist = [target] + varlist

        if time_vars != None:
            varlist = [i for i in varlist if i not in time_vars]
            varlist = varlist + time_vars
        if groupvars != None:
            varlist = [i for i in varlist if i not in groupvars]
            varlist = varlist + groupvars
        if exclude_vars != None:
            varlist = [i for i in varlist if i not in exclude_vars]

        for var in varlist:
            try:
                print(var)
                tmp_al = data.copy()
                tmp_df = data.copy()
                try:
                    if len(tmp_df[var].unique()) > 30:
                        cols = [var]
                        Q1 = tmp_df[cols].quantile(0.25)
                        Q3 = tmp_df[cols].quantile(0.75)
                        IQR = Q3 - Q1
                        tmp_df = tmp_df[
                            ~(
                                (tmp_df[cols] < (Q1 - 1.5 * IQR))
                                | (tmp_df[cols] > (Q3 + 1.5 * IQR))
                            ).any(axis=1)
                        ]
                    if len(tmp_df) < 5:
                        tmp_df = data.copy()
                except:
                    print('Error in removing outliers')
                tmp_df = tmp_df[tmp_df[target].notna()]
                tmp_df = tmp_df[tmp_df[var].notna()]
                tmp_al = tmp_al[tmp_al[target].notna()]
                tmp_al = tmp_al[tmp_al[var].notna()]
                len_al, len_tm = len(tmp_al), len(tmp_df)

                if len_tm < len_al:
                    llt = True
                    txt_outlier = ' (Outliers Removed)'
                else:
                    llt = False
                    txt_outlier = ' (All Data - No Outliers Present)'

                # plot1: Single var boxplot + histogram (uses 'var')
                if plot1_box_dist_all:
                    f, (ax_box, ax_hist) = plt.subplots(
                        2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=figsize
                    )
                    sns.boxplot(tmp_df[var].dropna(axis=0), ax=ax_box)
                    ax_box.set(xlabel=None, title=var + txt_outlier)
                    sns.distplot(tmp_df[var].dropna(axis=0), ax=ax_hist)
                    ax_hist.set(xlabel=' ')
                    pdf.savefig()
                    plt.close()
                    if llt == True:
                        f, (ax_box, ax_hist) = plt.subplots(
                            2,
                            sharex=True,
                            gridspec_kw={'height_ratios': (0.15, 0.85)},
                            figsize=figsize,
                        )
                        sns.boxplot(tmp_al[var].dropna(axis=0), ax=ax_box)
                        ax_box.set(xlabel=None, title=var + ' (All Data)')
                        sns.distplot(tmp_al[var].dropna(axis=0), ax=ax_hist)
                        ax_hist.set(xlabel=' ')
                        pdf.savefig()
                        plt.close()

                # plot2: Grouped boxplot (horizontal) + histogram (iterates through 'groupvars')
                if (plot2_box_dist_grp) & (groupvars != None):  ###
                    for groupvar in groupvars:
                        f, (ax_box, ax_hist) = plt.subplots(
                            2,
                            sharex=True,
                            gridspec_kw={'height_ratios': (0.35, 0.65)},
                            figsize=figsize,
                        )
                        sns.boxplot(
                            x=var,
                            y=groupvar,
                            data=tmp_df[[var, groupvar]].dropna(axis=0),
                            orient='h',
                            color='white',
                            ax=ax_box,
                        )
                        ax_box.set(
                            xlabel=None, title=var + txt_outlier + ' - Grouped by ' + groupvar
                        )
                        sns.histplot(
                            tmp_df[[var, groupvar]].dropna(axis=0),
                            x=var,
                            hue=groupvar,
                            alpha=0.5,
                            ax=ax_hist,
                        )
                        ax_hist.set(xlabel=' ')
                        pdf.savefig()
                        plt.close()
                        if llt == True:
                            f, (ax_box, ax_hist) = plt.subplots(
                                2,
                                sharex=True,
                                gridspec_kw={'height_ratios': (0.35, 0.65)},
                                figsize=figsize,
                            )
                            sns.boxplot(
                                x=var,
                                y=groupvar,
                                data=tmp_al[[var, groupvar]].dropna(axis=0),
                                orient='h',
                                color='white',
                                ax=ax_box,
                            )
                            ax_box.set(
                                xlabel=None, title=var + ' (All Data)' + ' - Grouped by ' + groupvar
                            )
                            sns.histplot(
                                tmp_al[[var, groupvar]].dropna(axis=0),
                                x=var,
                                hue=groupvar,
                                alpha=0.5,
                                ax=ax_hist,
                            )
                            ax_hist.set(xlabel=' ')
                            pdf.savefig()
                            plt.close()

                # plot3: Grouped boxplot (vertical) (iterates through 'groupvars')
                if (plot3_box_grp) & (groupvars != None):
                    for groupvar in groupvars:
                        f, (ax_1) = plt.subplots(1, figsize=figsize)
                        sns.boxplot(x=groupvar, y=var, data=tmp_df, hue=groupvar, ax=ax_1)
                        ax_1.set(title=var + txt_outlier + ' - Grouped by ' + groupvar)
                        pdf.savefig()
                        plt.close()
                        if llt == True:
                            f, (ax_1) = plt.subplots(1, figsize=figsize)
                            sns.boxplot(x=groupvar, y=var, data=tmp_al, hue=groupvar, ax=ax_1)
                            ax_1.set(title=var + ' (All Data)' + ' - Grouped by ' + groupvar)
                            pdf.savefig()
                            plt.close()

                # plot4: Double-grouped boxplot (vertical) (iterates through 'grouppairs')
                if (plot4_box_pairs_grp) & (grouppairs != None):
                    for groupvar in grouppairs:
                        f, (ax_1) = plt.subplots(1, figsize=figsize)
                        sns.boxplot(x=groupvar[0], y=var, data=tmp_df, hue=groupvar[1], ax=ax_1)
                        ax_1.set(
                            title=var
                            + txt_outlier
                            + ' - Grouped by '
                            + groupvar[0]
                            + ' & '
                            + groupvar[1]
                        )
                        pdf.savefig()
                        plt.close()
                        if llt == True:
                            f, (ax_1) = plt.subplots(1, figsize=figsize)
                            sns.boxplot(x=groupvar[0], y=var, data=tmp_al, hue=groupvar[1], ax=ax_1)
                            ax_1.set(
                                title=var
                                + ' (All Data)'
                                + ' - Grouped by '
                                + groupvar[0]
                                + ' & '
                                + groupvar[1]
                            )
                            pdf.savefig()
                            plt.close()

                # plot5: Regplot between var and target
                if (plot5_reg_all == True) & (var != target):
                    f, (ax_1) = plt.subplots(1, figsize=figsize)
                    sns.regplot(x=var, y=target, data=tmp_df, ax=ax_1)
                    ax_1.set(title=var + txt_outlier)
                    pdf.savefig()
                    plt.close()
                    if (llt == True) & (var != target):
                        f, (ax_1) = plt.subplots(1, figsize=figsize)
                        sns.regplot(x=var, y=target, data=tmp_al, ax=ax_1)
                        ax_1.set(title=var + ' (All Data)')
                        pdf.savefig()
                        plt.close()

                # plot6: Grouped scatterplot (hue=groupvar) between var and target (iterates through 'groupvars')
                if (plot6_scatter_grp == True) & (groupvars != None) & (var != target):
                    for groupvar in groupvars:
                        _plot_scatter(
                            tmp_df,
                            var,
                            target,
                            alpha=0.4,
                            hue=groupvar,
                            figsize=figsize,
                            outpdf=pdf,
                            title=var + txt_outlier + ' - Grouped by ' + groupvar,
                        )
                        if llt == True:
                            _plot_scatter(
                                tmp_al,
                                var,
                                target,
                                alpha=0.4,
                                hue=groupvar,
                                figsize=figsize,
                                outpdf=pdf,
                                title=var + ' (All Data)' + ' - Grouped by ' + groupvar,
                            )

                # plot7: Scatterplot (hue=None) between time_var and var
                if (plot7_scatter_time_all == True) & (time_vars != None):
                    for time_var in time_vars:
                        _plot_scatter(
                            tmp_df,
                            time_var,
                            var,
                            alpha=0.4,
                            hue=None,
                            figsize=figsize,
                            outpdf=pdf,
                            title=var + txt_outlier,
                        )
                        if llt == True:
                            _plot_scatter(
                                tmp_al,
                                time_var,
                                var,
                                alpha=0.4,
                                hue=None,
                                figsize=figsize,
                                outpdf=pdf,
                                title=var + ' (All Data)',
                            )

                # plot8: Grouped scatterplot (hue=groupvar) between time_var and var (iterates through 'groupvars')
                if (plot8_scatter_time_grp == True) & (groupvars != None) & (time_vars != None):
                    for groupvar in groupvars:
                        for time_var in time_vars:
                            _plot_scatter(
                                tmp_df,
                                time_var,
                                var,
                                alpha=0.4,
                                hue=groupvar,
                                figsize=figsize,
                                outpdf=pdf,
                                title=var + txt_outlier + ' - Grouped by ' + groupvar,
                            )
                            if llt == True:
                                _plot_scatter(
                                    tmp_al,
                                    time_var,
                                    var,
                                    alpha=0.4,
                                    hue=groupvar,
                                    figsize=figsize,
                                    outpdf=pdf,
                                    title=var + ' (All Data)' + ' - Grouped by ' + groupvar,
                                )
            except:
                pass


# df = pd.read_excel('./testdata_all_graphs.xlsx')


# all_graphs(data         = df,
#            target       = 'F96',
#            outfile_pdf  = './test_visuals01.pdf'
#            )

# all_graphs(data         = df,
#            target       = 'F96',
#            outfile_pdf  = './test_visuals02.pdf',
#            groupvars    = ['F3','F2','F16'],
#            exclude_vars = ['F2']
#            )

# all_graphs(data         = df,
#            target       = 'F96',
#            outfile_pdf  = './test_visuals03.pdf',
#            groupvars    = ['F3','F2','F16'],
#            grouppairs   = [('F3','F2'),('F3','F16')],
#            exclude_vars = ['F2']
#            )

# all_graphs(data         = df,
#            target       = 'F96',
#            outfile_pdf  = './test_visuals04.pdf',
#            time_vars    = 'B1',
#            groupvars    = ['F3','F2','F16'],
#            grouppairs   = [('F3','F2'),('F3','F16')],
#            exclude_vars = ['F2']
#            )

# all_graphs(data         = df,
#            target       = 'F96',
#            varlist      = ['F93','D49'],
#            outfile_pdf  = './test_visuals05.pdf',
#            time_vars    = 'B1',
#            groupvars    = ['F3','F2','F16'],
#            grouppairs   = [('F3','F2'),('F3','F16')],
#            exclude_vars = ['F2']
#            )

# all_graphs(data         = df,
#            target       = 'F96',
#            #varlist      = ['F93','D49'],
#            outfile_pdf  = './test_visuals06.pdf',
#            time_vars    = 'B1',
#            groupvars    = ['F3','F2','F16'],
#            grouppairs   = [('F3','F2'),('F3','F16')],
#            exclude_vars = ['F2'],
#            allnumvars   = True,
#            )
