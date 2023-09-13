# -*- coding: utf-8 -*-
"""
Created on July 1 19:56:28 2022

AUTHOR:     Kevin Baughman?

Plot rolling correlations for a set of variables and save the results as a pdf.

Rolling correlations are correlations between two time series on a rolling window. 
One benefit of this type of correlation is that you can visualize the correlation 
between two time series over time.

This funcion performs the following tasks:

1. Converts sort var to date and sorts dataframe
2. Computes mean, std, r, and abs(r) for all periods
3. Subsets to only the top N vars (num_corr_vars) by abs(r) 
4. Plots grid of stats for each var next to rolling corr series plot (all vars within each time window)
5. Plots series graph of rolling corr for each var grouped by time window
6. Creates one plot per var and time window combination
7. Outputs all plots to pdf file

PARAMETERS:

- data: dataframe
      Dataframe that must contain the variables, target and dates.
- targets: string
      Name of the target variable
- varlist: list of strings
      List of the names of the strings
- sortvar: string 
      Name of the variable that will sort the values (time stamps or any other time order)
- sort: list of strings
      List of the names of the strings
- sort_fmt: string (optional)
      Format of the datatime object that we want as a return
- periods: list of integers
      List of the sizes of the moving windows
- output: string (path)
      Path to the PDF where the output will be displayed.

      Recommendation: Use string literal concatenation for the path (e.g. r"./rolling_correlations.pdf")
- num_corr_vars: integer (optional)
      The number of correlated variables for which we will compute average correlations (typically len(varlist))
- graph_each_var: boolean (optional)
      Boolean for asking if we want the plot of all the variables individually
- graph_each_var_win: boolean (optional)
      Boolean for asking if we want the plot of all the variables windows individually
- graph_xlabel: string (optional)
      Name that we want to use as the plot's xlabel
- num_vars_graph: integer (optional)
      The number of variables to plot (typically len(varlist))
- num_vars_output: integer (optional)
      The number of variables that will be used to perform the output (typically len(varlist)).
- precision: integer (optional)
      The round of digits that we want to use to perform the rolling correlation
- sort_alphabetic: boolean (optional)
      Boolean for asking whether we want to sort the data alphabetically (only ascending).

OUTPUTS:    rolling_correlations.pdf

DEPENDENCY: None

NOTES:      generate_fake_dataframe func details: https://towardsdatascience.com/generating-fake-data-with-pandas-very-quickly-b99467d4c618
            
CHANGES:    WN - 03AUG2022- Added precision and sort_alphabetic params along with usage example on synthetic data


"""

#%% IMPORT NECESSARY PACKAGES ###

import seaborn as sns
import pandas as pd 
import numpy as np

from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#%% DEFINE FUNCTION ###

def rollingCorrs(
                 data,
                 targets,
                 varlist,
                 sortvar,
                 periods,
                 output='./rolling_corrs.pdf',
                 num_corr_vars=50,
                 sort=True,
                 sort_fmt='%Y-%m-%d',
                 graph_all_windows=True,
                 graph_each_var=True,
                 graph_each_var_win=False,
                 graph_xlabel=None,
                 graph_ylabel='Pearson Correlation Coef. (r)',
                 graph_x_fontsize=14,
                 graph_y_fontsize=14,
                 tabl_ratio='r',
                 tabl_ratio_abs='abs_r',
                 tabl_fontsize=12,
                 num_vars_graph=None,
                 num_vars_output=None,
                 precision=3,
                 sort_alphabetic=False
                 ):

    # Create lists for all iterable parms
    if type(targets) != list:
        targets = [targets]
    if varlist == None:
        varlist = [i for i in data.columns.values if i not in targets]
    if type(varlist) != list:
        varlist = [varlist]
    if type(periods) != list:
        periods = [periods]

    # Apply default vals if needed
    if graph_xlabel == None:
        graph_xlabel = sortvar
    if num_vars_graph == None:
        num_vars_graph = 5
    if num_vars_output == None:
        num_vars_output = 10000

    # Sort df if sort is True
    if sort:
        data = data.sort_values(sortvar)

    # Create date only variable (remove time component)
    data['_dt_'] = pd.to_datetime(
        data[sortvar].dt.strftime(sort_fmt), errors='coerce')

    # Iterate and create rolling correlation values
    stacked = pd.DataFrame()
    for target in targets:
        tmp1 = data[['_dt_', target] + varlist]
        for period in periods:
            # tmp2  = tmp1[tmp1['_dt_'] <= tp]
            dt = ['_dt_', target]
            # .dropna(how='any', axis=0)
            tmp2 = tmp1[dt + varlist].reset_index(drop=True)
            dfcor = tmp2[target].rolling(period).corr(tmp2[varlist])
            tmp2 = tmp2[dt].join(dfcor)
            tmp2['_target_'], tmp2['_window_'] = target, period
            tmp2 = tmp2.set_index(['_target_', '_window_'])
            stacked = stacked.append(tmp2)

    # Distinct list of 'target' vals
    tvals = list(set(stacked.index.get_level_values('_target_')))
    # Distinct list of 'window' vals
    wvals = sorted(set(stacked.index.get_level_values('_window_')))
    # Total num of windows (to be used as a check in later PDF processing)
    lenwvals = len(wvals)

    with PdfPages(output) as pdf:
        for t in tvals:
            # Prep dataset for specific target variable
            # d1 = d1[[target, '_dt_'] + varlist]
            d1 = stacked[stacked.index.get_level_values('_target_').isin([t])]

            if graph_all_windows == True:
                d2 = d1[~d1['_dt_'].isnull()]
                mn = d2.drop(target, 1).mean()
                sd = d2.drop(target, 1).std()
                sr = pd.concat([mn, sd], axis=1).rename(columns={0: 'mean', 1: 'std'})
                sr[tabl_ratio] = sr['mean'] / sr['std']
                sr[tabl_ratio_abs] = abs(sr[tabl_ratio])
                sr = sr.sort_values(tabl_ratio_abs, ascending=False).dropna(how='any', axis=0)

                sr = sr.astype(float).round(precision)
                if sort_alphabetic == True:
                    sr = sr.sort_index(ascending=True)
                print(sr)

                if len(sr.index) < num_corr_vars:
                    txt_table = 'Avg. Correlations with ' + t
                    lensr = len(sr.index)
                else:
                    txt_table = 'Avg. Correlations with ' + t + ' (Top ' + str(num_corr_vars) + ')'
                    lensr = num_corr_vars
                topvars = list(sr.index[:num_corr_vars])
                topvars_all = topvars.copy()
                sr.index.name = 'Var'
                sr = sr.reset_index()

                # Fig 1: First page - table on left and graph of all windows on right
                fig = plt.figure(figsize=(14, 7))
                grid = plt.GridSpec(6, 6, hspace=0.9, wspace=0.45)
                y_ts = fig.add_subplot(grid[::, :-4])
                main_ax2 = fig.add_subplot(grid[::, 2:])

                # Fig 1 (Table 1)
                y_ts.axis('off')
                tabl = y_ts.table(cellText=sr.round(3).values[:lensr],
                                  colLabels=sr.columns,
                                  loc='center',
                                  bbox=[0, -.1, 1, 0.99]
                                  )
                tabl.auto_set_font_size(True)
                tabl.set_fontsize(tabl_fontsize)
                y_ts.set_title('Avg. Correlations with ' + t, y=0.92, fontsize=14)

                # Fig 1 (Plot 2): Rolling correlations - all windows
                d2[['_dt_'] + topvars[:num_vars_graph]].set_index('_dt_').plot(ax=main_ax2)
                plt.xlabel(graph_xlabel, fontsize=graph_x_fontsize)
                plt.title('Rolling Correlations: ' + t + ' & Top ' + str(num_vars_graph) + ' Parameters' + 'All Windows Plotted', fontsize=15)

                pdf.savefig()

            # If more than one window in wvals then output same fig as above, but for each window
            if lenwvals > 1:
                for win in wvals:
                    d2 = d1[~d1['_dt_'].isnull()]
                    d2 = d2[d2.index.get_level_values('_window_').isin([win])]
                    mn = d2.drop(target, 1).mean()
                    sd = d2.drop(target, 1).std()
                    sr = pd.concat([mn, sd], axis=1).rename(columns={0: 'mean', 1: 'std'})
                    sr[tabl_ratio] = sr['mean'] / sr['std']
                    sr[tabl_ratio_abs] = abs(sr[tabl_ratio])
                    sr = sr.sort_values(tabl_ratio_abs, ascending=False).dropna(how='any', axis=0)

                    sr = sr.astype(float).round(precision)
                    if sort_alphabetic == True:
                        sr = sr.sort_index(ascending=True)
                    print(sr)
                    
                    if len(sr.index) < num_corr_vars:
                        txt_table = 'Avg. Correlations with ' + t
                        lensr = len(sr.index)
                    else:
                        txt_table = 'Avg. Correlations with ' + t + ' (Top ' + str(num_corr_vars) + ')'
                        lensr = num_corr_vars
                    topvars = list(sr.index[:num_corr_vars])
                    sr.index.name = 'Var'
                    sr = sr.reset_index()

                    # Fig 2-X: After first page - table on left and graph of each window on right
                    fig = plt.figure(figsize=(14, 7))
                    grid = plt.GridSpec(6, 6, hspace=0.9, wspace=0.45)
                    y_ts = fig.add_subplot(grid[::, :-4])
                    main_ax2 = fig.add_subplot(grid[::, 2:])

                    # Fig 2-X (Table)
                    y_ts.axis('off')
                    tabl = y_ts.table(cellText=sr.round(3).values[:lensr],
                                      colLabels=sr.columns,
                                      loc='center',
                                      bbox=[0, -.1, 1, 0.99]
                                      )
                    tabl.auto_set_font_size(True)
                    tabl.set_fontsize(tabl_fontsize)
                    y_ts.set_title('Avg. Correlations with ' + t, y=0.92, fontsize=14)

                    # Fig 2-X (Plot): Rolling correlations - each window
                    d2[['_dt_'] + topvars[:num_vars_graph]].set_index('_dt_').plot(ax=main_ax2)
                    plt.xlabel(graph_xlabel, fontsize=graph_x_fontsize)
                    plt.title('Rolling Correlations: ' + t + ' & Top ' + str(num_vars_graph) + ' Parameters' + '\n' + 'Window: ' + str(win) + ' obs', fontsize=15)

                    pdf.savefig()

            # Iterate over available process parms
            if graph_each_var == True:
                if num_vars_output != None:
                    topvars_all = topvars_all[:num_vars_output]
                for v in topvars_all:  # [:num_vars_graph]:
                    tmpv = d1[~d1['_dt_'].isnull()].reset_index().rename(columns={'_window_': 'Window'})
                    tmpv['No Correlation'] = 0.0
                    f, (ax_1) = plt.subplots(1, figsize=(14, 7))
                    sns.lineplot(data=tmpv, x="_dt_", y=v, hue="Window", ax=ax_1)
                    ax_1.axhline(0, ls='--', color='lightgray')
                    plt.title(v + ': Rolling Correlations (Grouped by Window Size)', fontsize=15)
                    plt.ylabel(graph_ylabel, fontsize=graph_y_fontsize)
                    plt.xlabel(graph_xlabel, fontsize=graph_x_fontsize)
                    pdf.savefig()

                    if lenwvals > 1 and graph_each_var_win == True:
                        for vw in sorted(wvals):
                            # Prep dataset for individual var and window graph
                            tmpv = d1[~d1['_dt_'].isnull()]
                            tmpv = tmpv[tmpv.index.get_level_values('_window_').isin([vw])]
                            tmpv['Mean'] = tmpv[v].mean()
                            tmpv['No Correlation'] = 0.0
                            tmpv = tmpv[['_dt_'] + [v, 'Mean', 'No Correlation']].set_index('_dt_').dropna(how='any')
                            # Plot rolling correlation for one var and one window size
                            if len(tmpv) > 0:
                                tmpv.plot(figsize=(14, 7), color=['b', 'orange', 'lightgray'])
                                plt.title('Rolling Correlation (Window: ' + str(vw) + ' obs): ' + t + ' & ' + v, fontsize=15)
                                plt.xlabel(graph_xlabel, fontsize=graph_x_fontsize)
                                plt.ylabel(graph_ylabel, fontsize=graph_y_fontsize)
                                pdf.savefig()
    return stacked


#%% EXECUTE FUNCTION - EXAMPLE FROM GRIFOLS ###

# rollingCorrs(
#              data = df,
#              targets = 'F96',
#              varlist = ['F43', 'F68', 'F90'],
#              sortvar = 'date',
#              sort=True,
#              sort_fmt='%Y-%m-%d',
#              periods = [90, 180, 365],
#              output = r"./acs-grifols-la/exploration/rolling_correlations.pdf",
#              num_corr_vars = 3,
#              graph_each_var = True,
#              graph_each_var_win = True,
#              graph_xlabel = 'F25: Date',
#              num_vars_graph = 3,
#              num_vars_output = 3,
#              precision = 4,
#              sort_alphabetic = True
#              )

#%% TEST FUNCTION ON SYNTHETIC DATA ###

from itertools import cycle
# first generate synthetic dataframe
# See here for usage details: https://towardsdatascience.com/generating-fake-data-with-pandas-very-quickly-b99467d4c618
def generate_fake_dataframe(size, cols, col_names = None, intervals = None, seed = None):
    
    categories_dict = {'animals': ['cow', 'rabbit', 'duck', 'shrimp', 'pig', 'goat', 'crab', 'deer', 'bee', 'sheep', 'fish', 'turkey', 'dove', 'chicken', 'horse'],
                       'names'  : ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth', 'Ahmed', 'Barbara', 'Richard', 'Susan', 'Salomon', 'Juan Luis'],
                       'cities' : ['Stockholm', 'Denver', 'Moscow', 'Marseille', 'Palermo', 'Tokyo', 'Lisbon', 'Oslo', 'Nairobi', 'Río de Janeiro', 'Berlin', 'Bogotá', 'Manila', 'Madrid', 'Milwaukee'],
                       'colors' : ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'pink', 'silver', 'gold', 'beige', 'brown', 'grey', 'black', 'white']
                      }
    default_intervals = {"i" : (0,10), "f" : (0,100), "c" : ("names", 5), "d" : ("2020-01-01","2020-12-31")}
    rng = np.random.default_rng(seed)

    first_c = default_intervals["c"][0]
    categories_names = cycle([first_c] + [c for c in categories_dict.keys() if c != first_c])
    default_intervals["c"] = (categories_names, default_intervals["c"][1])
    
    if isinstance(col_names,list):
        assert len(col_names) == len(cols), f"The fake DataFrame should have {len(cols)} columns but col_names is a list with {len(col_names)} elements"
    elif col_names is None:
        suffix = {"c" : "cat", "i" : "int", "f" : "float", "d" : "date"}
        col_names = [f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)]

    if isinstance(intervals,list):
        assert len(intervals) == len(cols), f"The fake DataFrame should have {len(cols)} columns but intervals is a list with {len(intervals)} elements"
    else:
        if isinstance(intervals,dict):
            assert len(set(intervals.keys()) - set(default_intervals.keys())) == 0, f"The intervals parameter has invalid keys"
            default_intervals.update(intervals)
        intervals = [default_intervals[col] for col in cols]
    df = pd.DataFrame()
    for col, col_name, interval in zip(cols, col_names, intervals):
        if interval is None:
            interval = default_intervals[col]
        assert (len(interval) == 2 and isinstance(interval, tuple)) or isinstance(interval, list), f"This interval {interval} is neither a tuple of two elements nor a list of strings."
        if col in ("i","f","d"):
            start, end = interval
        if col == "i":
            df[col_name] = rng.integers(start, end, size)
        elif col == "f":
            df[col_name] = rng.uniform(start, end, size)
        elif col == "c":
            if isinstance(interval, list):
                categories = np.array(interval)
            else:
                cat_family, length = interval
                if isinstance(cat_family, cycle):
                    cat_family = next(cat_family)
                assert cat_family in categories_dict.keys(), f"There are no samples for category '{cat_family}'. Consider passing a list of samples or use one of the available categories: {categories_dict.keys()}"
                categories = rng.choice(categories_dict[cat_family], length, replace = False, shuffle = True)
            df[col_name] = rng.choice(categories, size, shuffle = True)
        elif col == "d":
            df[col_name] = rng.choice(pd.date_range(start, end), size)
    return df       

"""
df_synth = generate_fake_dataframe(
              size = 300, 
              cols = "dcifff", 
              col_names = ["D1", "C1", "T1", "V1", "V2", "V3"],
              intervals = [("2020-01-01","2020-12-31"), ("cities", 15), (0,100), (0.00,0.99), (1.0,10.0), (1000.0,9999.9)],
              seed = 12345
              )

#%% Now apply rollingcorr func on synthetic data ###

rollingCorrs(
             data = df_synth,
             targets = 'T1',
             varlist = ['V1', 'V2', 'V3'],
             sortvar = 'D1',
             sort=True,
             sort_fmt='%Y-%m-%d',
             periods = [30, 60, 90],
             output = r"./rolling_correlations.pdf",
             num_corr_vars = 3,
             graph_each_var = True,
             graph_each_var_win = True,
             graph_xlabel = 'D1: Date',
             num_vars_graph = 3,
             num_vars_output = 3,
             precision = 4,
             sort_alphabetic = True
             )

"""
#%%
