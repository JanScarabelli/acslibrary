#!/usr/bin/env python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import datetime


def create_optimal_bins(
    data,
    output,
    target_var,
    tree_vars=[],
    min_obs=50,
    tree_depths=3,
    group_vars=[],
    remove_outliers=False,
):
    """
    Generate pdf with decision tree binning
    Parameters:
    -------------
    data: dataframe
    output: string
        Where to generate the pdf
    target_var: string
    tree_vars: list
        parameters to split against the target
    min_obs: int
        minimum observation per leaf
    tree_depths: int
    group_vars: list
        categorical variable to split dataset
    remove_outliers: boolean

    Returns
    ------------
    generates a pdf
    """
    grp = False
    if group_vars != []:
        grp = True

    if type(tree_vars) != list:
        tree_vars = [tree_vars]
    if type(min_obs) != list:
        min_obs = [min_obs]
    if type(tree_depths) != list:
        tree_depths = [tree_depths]
    if type(group_vars) != list:
        group_vars = [group_vars]
    if type(remove_outliers) != list:
        remove_outliers = [remove_outliers]

    # Begin outputing to PDF
    with PdfPages(output) as pdf:
        # Iterate over input vars (parm: tree_vars)
        for _var in tree_vars:
            try:
                if grp != True:
                    group_vars, list_of_cats = ['All'], ['All']
                # Iterate over group_vars (list of categorical variables)
                for group_var in group_vars:
                    if grp == True:
                        list_of_cats = ['__ALL__'] + list(set(data[group_var]))
                    # Iterate over categories from 'group_var'
                    for cat in list_of_cats:
                        # Iterate over outlier options (parm: remove_outliers)
                        for out in remove_outliers:
                            if grp == True:
                                tmp = data[[target_var, _var, group_var]].dropna()
                                if cat not in ['__ALL__', 'All']:
                                    tmp = tmp[tmp[group_var] == cat]
                                else:
                                    cat = 'All'
                            else:
                                tmp = data[[target_var, _var]].dropna()

                            outlier_txt = ''
                            if out != False:
                                try:
                                    # Remove outliers if specified (need a min # of obs)
                                    if len(tmp[_var].unique()) > 2:
                                        cols = [_var]
                                        Q1 = tmp[cols].quantile(0.25)
                                        Q3 = tmp[cols].quantile(0.75)
                                        IQR = Q3 - Q1
                                        tmp = tmp[
                                            ~(
                                                (tmp[cols] < (Q1 - 1.5 * IQR))
                                                | (tmp[cols] > (Q3 + 1.5 * IQR))
                                            ).any(axis=1)
                                        ]
                                        outlier_txt = '   (Outliers Removed)'
                                    else:
                                        print('')
                                except:
                                    print('Error in removing outliers')

                            X = tmp[[_var]]  # Input var for decision tree regressor
                            y = tmp[[target_var]]  # Target var for decision tree regressor
                            # Iterate over list of tree_depths
                            for _tree_depth in tree_depths:
                                # Iterate over list of min_obs
                                for _min_obs_per_leaf in min_obs:
                                    # Tree parms
                                    print(
                                        'Var: '
                                        + str(_var)
                                        + '  Outliers: '
                                        + str(out)
                                        + '  Tree Depth: '
                                        + str(_tree_depth)
                                        + '   Min Obs: '
                                        + str(_min_obs_per_leaf)
                                    )
                                    clf = DecisionTreeRegressor(
                                        random_state=1234,
                                        max_depth=_tree_depth,
                                        min_samples_leaf=_min_obs_per_leaf,
                                    )
                                    model = clf.fit(X, y)

                                    # Begin plotting
                                    fig = plt.figure(figsize=(14, 7))
                                    grid = plt.GridSpec(6, 6, hspace=0.9, wspace=0.1)
                                    main_ax = fig.add_subplot(grid[:-3, :-4])
                                    y_ts = fig.add_subplot(grid[-3:, :-4])
                                    main_ax2 = fig.add_subplot(grid[::, 2:])

                                    p = tree.plot_tree(
                                        clf,
                                        feature_names=X.columns.values,
                                        filled=True,
                                        ax=main_ax2,
                                    )
                                    plt.title(
                                        'Tree Depth: '
                                        + str(_tree_depth)
                                        + '   Min Obs Per Leaf: '
                                        + str(_min_obs_per_leaf)
                                        + outlier_txt
                                    )

                                    n = tmp.copy()
                                    n['p'] = model.predict(X).round(3)

                                    nn = n.copy()
                                    nn = (
                                        nn.groupby('p')
                                        .min()[[_var]]
                                        .rename(columns={_var: 'Min Value'})
                                        .round(3)
                                        .join(
                                            nn.groupby('p')
                                            .max()[[_var]]
                                            .round(3)
                                            .rename(columns={_var: 'Max Value'})
                                        )
                                        .sort_index(ascending=False)
                                        .head(5)
                                    )
                                    nn.index.name = 'Prediction'
                                    nn = nn.reset_index()

                                    y_ts.axis('tight')
                                    y_ts.axis('off')
                                    tabl = y_ts.table(
                                        cellText=nn.values[:5],
                                        colLabels=nn.columns,
                                        loc='center',
                                        bbox=[0, -0.1, 1, 0.9],
                                    )
                                    tabl.auto_set_font_size(False)
                                    tabl.set_fontsize(12)
                                    y_ts.set_title('Top 5 Bins', y=0.83)

                                    sns.scatterplot(
                                        x=X.columns.values[0],
                                        y=target_var,
                                        data=n,
                                        hue='p',
                                        legend=False,
                                        ax=main_ax,
                                    )
                                    fig.suptitle(
                                        target_var + ' vs. ' + _var + ' (Group=' + cat + ')',
                                        x=0.5,
                                        y=0.97,
                                        fontweight='bold',
                                    )
                                    main_ax.set_title(
                                        'Darker colors => Higher Prediction Values',
                                        y=0.982,
                                        fontsize=11,
                                        style='italic',
                                    )
                                    pdf.savefig()
            except:
                print('Failed to save tree to PDF')


# createOptimalBins(data        = df,
#                   output      = 'temp.pdf',
#                   target_var  = 'F96',
#                   tree_vars   = ['C33','B14','D19'],
#                   min_obs     = 100,
#                   tree_depths = 2,
#                   group_vars  = [],
#                   remove_outliers = False
#                   )
