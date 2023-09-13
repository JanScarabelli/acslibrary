import pickle
import json
import os
from re import A
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from regex import F
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import string
from sklearn.neighbors import LocalOutlierFactor


from acslibrary.plots.heatmaps import plot_clustermap_with_left_bar, plot_heatmap_with_left_bar
from acslibrary.plots.tables import plot_dataframe
from acslibrary.plots.boxplots import draw_boxplots
from acslibrary.plots.timeseries import plot_scada_heatmap_with_left_bar, plot_profile
from acslibrary.plots.waterfall import waterfall_plot
from acslibrary.methods.outlier_removal import IQR_topN
from acslibrary.methods.anova_test import ANOVA
from acslibrary.methods.regression_analysis import plot_regression_analysis

def std_build_pipe(n_clusters, random_state=2):
    return Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'kmeans',
                KMeans(n_clusters=n_clusters, max_iter=1000, n_init=100, random_state=random_state),
            ),
        ]
    )

def order_clusters(data, cluster_var, order_var, letters=True):
    avgs = data.groupby(cluster_var)[order_var].mean().sort_values().reset_index()

    if letters:
        letters = list(string.ascii_uppercase)
        avgs['new_labels'] = list(reversed(letters[0 : avgs.shape[0]]))
    else:
        avgs['new_labels'] = list(range(avgs.shape[0]))

    avgs = avgs.drop(order_var, axis=1).set_index(cluster_var)
    to_replace = avgs.to_dict()['new_labels']

    data[cluster_var] = data[cluster_var].replace(to_replace)
    return data, to_replace

def outliers_detection_timeseries(data):

    clf = LocalOutlierFactor(n_neighbors=data.shape[0]-10)
    predictions = clf.fit_predict(data)

    return [batch for batch, pred in zip(data.index.values.tolist(), predictions) if pred == -1]

def plot_clustering_metrics(
    n, vec, build_pipe_func=std_build_pipe, outer_spec=None, fig=None, pdf_obj=None
):

    if outer_spec is not None:
        inner_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec)
        a0 = plt.Subplot(fig, inner_spec[0])
        a1 = plt.Subplot(fig, inner_spec[1])
    else:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 5))

    sse = {}
    ecv = {}
    for k in range(2, n + 1):
        kmeans = build_pipe_func(k).fit(vec)
        ecv[k] = kmeans['kmeans'].inertia_
        clusters = kmeans.predict(vec)
        sse[k] = silhouette_score(vec, clusters, random_state=3294)

    sns.lineplot(x=list(sse.keys()), y=list(sse.values()), ax=a0)
    a0.set_xlabel('Number of Clusters')
    a0.set_ylabel('Silhouette')
    a0.set_title('Silhouette Scores')

    sns.lineplot(x=list(ecv.keys()), y=list(ecv.values()), ax=a1)
    a1.set_xlabel('Number of Clusters')
    a1.set_ylabel('Within Clusters Sum of Squares')
    a1.set_title('Elbow Curve')

    if outer_spec is not None:
        fig.add_subplot(a0)
        fig.add_subplot(a1)

    if pdf_obj is not None:
        pdf_obj.savefig()

    return (a0, a1)

def outliers_detection(data):

    clf = LocalOutlierFactor(n_neighbors=data.shape[0]-10)
    predictions = clf.fit_predict(data)

    return [batch for batch, pred in zip(data.index.values.tolist(), predictions) if pred == -1]

def cluster_params(
    var_list,
    data,
    targets,
    sort_var,
    cat_vars=[],
    var_detail=False,
    max_clusters=5,
    min_clusters=2,
    rmv_out=None,
    starting_date=None,
    ending_date=None,
    show_metrics=True,
    tukey=False,
    normalize=True,
    build_pipe_func=std_build_pipe,
    to_letters=False,
    to_pdf=True,
    path=''
):
    """
    This function performs clustering analysis on the input data to identify clusters of similar data points. The resulting
    clusters are then plotted on a scatter plot over time, as well as on a heatmap where the distribution of the input variables
    can be visualized. Moreover, the results of the ANOVA test conducted are shown in the form of a boxplot and a table.

    Parameters:
    ------------

    var_list: list of strings
      The names of the variables to be used for clustering

    data: array-like
      The input data

    targets: list of strings
      The names of the variables to compare clusters to

    sort_var: string
      Variable to be used to sort batches in scatter plot (timestamp or timedelta works perfectly)

    cat_vars: list of strings
      The names of the categorical variables to compare clusters to (default is an empty list)

    var_detail: boolean
      Whether to plot a histogram and print statistics for each variable in `var_list` (default is False)

    max_clusters: int
      The maximum number of clusters to test (default is 5)

    min_clusters: int
      The minimum number of clusters to test (default is 2)

    rmv_out: int or None
      Number of outliers to remove using IQR and if it is not None, apply Local Outlier Factor (LOF). If no outliers are to be removed, set it to None (default is None)

    starting_date: timestamp or None
      The starting date for data used in analysis (clustering models will be created with the whole dataset, irrespective of this parameter). Only effective if `sort_var` is a timestamp (default is None)

    ending_date: timestamp or None
      The ending date for data used in analysis. Only effective if `sort_var` is a timestamp (default is None)

    show_metrics: boolean
      Thether to show the elbow and silhouette plots (default is True). It may take a while to produced those, since silhouette calculations contain random elements and to ensure consistent scores, we run 1000 initializations per model

    Tukey: boolean
      Whether to perform Tukey tests for categorical variables. Only effective if `cat_vars` is not empty (default is False)

    Normalize: boolean
      Whether to normalize values in the heatmaps (default is True)

    build_pipe_func: function
    A function to build the clustering model pipeline (default is `std_build_pipe`)

    to_letters: boolean
      whether to transform cluster labels into letters, with Cluster A being the one with the highest `targets[0]` (default is False)

    to_pdf: boolean
      Whether to save all output in a PDF file (default is True)

    path: string
      Where to store the PDF (default is an empty string)

    Returns:
    --------

    The input dataframe with additional columns containing the cluster labels 
    """

    working_data = data[var_list + cat_vars + [sort_var] + targets].copy()

    if rmv_out:
        cldata = IQR_topN(working_data[var_list], rmv_out)
        outliers = outliers_detection(cldata)
        cldata = cldata.drop(outliers).copy()
        working_data = working_data.drop(outliers).copy()
    else: cldata = working_data[var_list]

    working_data = pd.merge(
        left=working_data[[sort_var] + targets + cat_vars],
        right=cldata,
        left_index=True,
        right_index=True,
        how='inner',
    )
    cluster_var_names = []

    if min_clusters != max_clusters:
        n_clusters = range(min_clusters, max_clusters + 1)
    else:
        n_clusters = [max_clusters]

    for k in n_clusters:
        var_name = f'clusters_{k}'
        model = build_pipe_func(k).fit(cldata)
        working_data[var_name] = model.predict(cldata)
        working_data, to_replace = order_clusters(working_data, var_name, targets[0], letters=to_letters)
        cluster_var_names += [var_name]

    if to_pdf:
        f = path
        with PdfPages(f) as pdf_obj:

            if show_metrics:
                plot_clustering_metrics(15, cldata, build_pipe_func, pdf_obj=pdf_obj)

            if starting_date is not None:
                analysis_data = working_data[working_data[sort_var] >= starting_date].copy()
            elif ending_date is not None:
                analysis_data = working_data[working_data[sort_var] <= ending_date].copy()
            else:
                analysis_data = working_data.copy()

            for k in n_clusters:

                fig = plt.figure(figsize=(25, 10))
                outer = gridspec.GridSpec(1, 2)
                inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
                ax = plt.Subplot(fig, inner[0])
            
                sns.scatterplot(
                    data = working_data, x = sort_var, y = f'clusters_{k}', alpha=0.1, ax=ax
                )
                ax.invert_yaxis()
                ax.set_title('Distribution of Clusters over Time')
                fig.add_subplot(ax)

                plot_heatmap_with_left_bar(
                    data=working_data,
                    var=var_list,
                    sort_batch_by=[f'clusters_{k}', targets[0]],
                    left_bar_CQA=[f'clusters_{k}'] + targets,
                    l_color_norm_by_cluster=False,
                    vmax=None,
                    vmin=None,
                    title=f'Clustering (K-means, {k} clusters) - Sorted by cluster and {targets[0]}',
                    outer_spec=outer[1],
                    fig=fig,
                    pdf_obj=pdf_obj,
                    normalize=normalize,
                )

                waterfall_plot(
                    data=working_data,
                    var=f'clusters_{k}',
                    sort_var=sort_var,
                    freq='M',
                    pdf_obj=pdf_obj,
                )

                ordered_clusters = np.sort(working_data[f'clusters_{k}'].unique())
                # plot targets boxplots
                for i in targets:
                    ad = analysis_data[~analysis_data[i].isna()]
                    
                    if ad[f'clusters_{k}'].nunique() > 1:
                        ANOVA(data=ad, cat1=f'clusters_{k}', target=i, pdf_obj=pdf_obj)

                    for c in cat_vars:
                        add = ad[~ad[c].isna()]
                        draw_boxplots(
                            add,
                            [i],
                            c,
                            f'clusters_{k}',
                            order=ordered_clusters,
                            pdf_obj=pdf_obj,
                            split_x=True,
                            pal='Set2',
                            fig_size=(30, 10),
                        )

                        a = (
                            add.groupby([f'clusters_{k}', c])[i]
                            .describe()[['count', 'mean', 'std']]
                            .reset_index()
                        )
                        plot_dataframe(a, pdf_obj=pdf_obj, fig_size=(10, 4))
           
                        ANOVA(
                            data=add,
                            cat1=f'clusters_{k}',
                            cat2=c,
                            target=i,
                            pdf_obj=pdf_obj,
                            do_tukey=tukey,
                        ) if tukey == True else None

                cats = cat_vars + [f'clusters_{k}'] if cat_vars else [f'clusters_{k}']
                plot_regression_analysis(
                    data=analysis_data.dropna(),
                    target=targets[0],
                    year_bias=sort_var,
                    categorical_bias=cats,
                    pdf=pdf_obj,
                )

                # plot boxplots for each input variable
                if var_detail:

                    n = len(var_list)

                    def plot_by_index(i0, iN):
                        sub_var_list = var_list[i0:iN]
                        draw_boxplots(
                            data=analysis_data,
                            Y=sub_var_list,
                            x=f'clusters_{k}',
                            category=None,
                            order=ordered_clusters,
                            pdf_obj=pdf_obj,
                            pal='Set2',
                        )

                        ntabs = len(cat_vars) + len(sub_var_list) + 1
                        nrows = len(cat_vars) + 2
                        fig = plt.figure(figsize=(20, ntabs * 5))
                        outer = gridspec.GridSpec(nrows, 1)

                        a = (
                            analysis_data.groupby([f'clusters_{k}'])[sub_var_list]
                            .mean()
                            .reset_index()
                        )
                        plot_dataframe(a, pdf_obj=None, outer_spec=outer[0], fig=fig)

                        for i in range(len(cat_vars)):
                            c = cat_vars[i]
                            a = (
                                analysis_data.groupby([f'clusters_{k}', c])[sub_var_list]
                                .mean()
                                .reset_index()
                            )
                            plot_dataframe(a, pdf_obj=None, outer_spec=outer[i + 1], fig=fig)

                        inner = gridspec.GridSpecFromSubplotSpec(
                            1, len(sub_var_list), subplot_spec=outer[nrows - 1]
                        )

                        for i in range(len(sub_var_list)):
                            v = sub_var_list[i]
                            ax = plt.Subplot(fig, inner[i])
                            sns.histplot(data=analysis_data[sub_var_list], x=v, ax=ax)
                            fig.add_subplot(ax)

                        pdf_obj.savefig()

                    if n > 5:
                        inxs = np.arange(0, n + 1, 5, dtype=int)
                        i = 0
                        while i < len(inxs) - 1:
                            plot_by_index(inxs[i], inxs[i + 1])
                            i += 1
                    else:
                        plot_by_index(0, n)
            
    plt.close('all')
    return working_data


def time_series_clustering(
    tags,
    agg_data,
    sort_var,
    targets,
    cat_vars=[],
    agg_tags=None,
    max_clusters=6,
    min_clusters=2,
    rmv_out=3,
    prefix='',
    suffix='',
    save_pmml=False,
    build_pipe_func=std_build_pipe,
    show_metrics=True,
    full=True,
    print_profiles = True,
    profiles_by_cat = True
):
    """
    Function that produces two files:
    - file doing the clusteirng, and showing the target variables distribution per cluster
    - file visualising the scada time series grouped by cluster

    scada_tags: dict
        example provided at the end of this file
    agg_data: dataframe
        aggregated data
    path: str
        path where we want to store the files generated
    prefix: str
        name of hte prefix to use when naming the models and the files
    suffix: str
        suffix to append at the end of hte file name. can be used to version files created. i.e. 'v_0'
    DL: DataLoader instance
    sort_var: str
        DateTime variable to sort data
    targets: str or list
        used as targets in the analysis
    cat_vars: str or list
        i.e. plasma cluster (categorical variable)
    max_clusters: int
    min_clusters: int
    debug=False
        set tpo True if you want to do some prints when it's failing
    """

    data_eng = agg_data.copy()
    var_names = []

    for t in tags.keys():
        if tags[t]['to_cluster']:
            dt = tags[t]['data'].copy()
            var_names += list(dt.columns)
            data_eng = data_eng.merge(right=dt, left_index=True, right_index=True, how='inner')

    folder_name = f'{prefix}_clustering_{suffix}'

    if agg_tags is not None:
        var_names += agg_tags
        var_detail = agg_tags
    else:
        var_detail = None
    
    ppc_data = cluster_params(
        var_list=var_names,
        data=data_eng,
        path=folder_name,
        targets=targets,
        sort_var=sort_var,
        cat_vars=cat_vars,
        var_detail=var_detail,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        rmv_out=rmv_out,
        prefix=prefix,
        suffix=suffix,
        show_metrics=show_metrics,
        save_pmml=save_pmml,
        build_pipe_func=build_pipe_func,
        to_pdf=full,
        time_series=True
    )

    for i in range(min_clusters, max_clusters + 1):
        ppc_data[f'{prefix}_{i}_clusters'] = ppc_data[f'{folder_name}_{i}_clusters'].astype(float)

    if print_profiles:
        with PdfPages(os.path.join(folder_name + '_models', f'{folder_name}_timeseries.pdf')) as pdf:

            for t in tags.keys():
                if tags[t]['to_cluster'] == True:
                    plot_scada_heatmap_with_left_bar(
                        tag_data=tags[t]['data'],
                        agg_data=ppc_data,
                        sort_batch_by=[sort_var],
                        sort_var=sort_var,
                        left_bar_CQA=targets,
                        title= t,
                        pdf=pdf,
                        vmax=None,
                        vmin=None,
                        val_cnt=False,
                    )

            for i in range(min_clusters, max_clusters + 1):
                for t in tags.keys():

                    c = f'{prefix}_{i}_clusters'
                    plot_scada_heatmap_with_left_bar(
                        tag_data=tags[t]['data'],
                        agg_data=ppc_data,
                        sort_batch_by=[c],
                        sort_var=sort_var,
                        left_bar_CQA=targets + [c],
                        title= t,
                        pdf=pdf,
                        vmax=None,
                        vmin=None,
                        val_cnt=False,
                    )

                    cs = ppc_data[c].sort_values().unique()
                    mean_profile = plot_profile(
                        tag_data=tags[t]['data'],
                        agg_data=ppc_data,
                        cat_var=c,
                        figsize=(20, 20),
                        ylabel=t,
                        pdf=pdf,
                        ylim=tags[t]['ylim'],
                        return_mean_profile=cs,
                    )

                    plot_dataframe(mean_profile.groupby(c).mean_profile.describe(), pdf_obj=pdf)

                    if profiles_by_cat:
                        
                        for cat_var in cat_vars:
                            for p_cat in ppc_data[cat_var].unique():
                                sub_data = ppc_data[ppc_data[cat_var] == p_cat]

                                plot_profile(
                                tag_data=tags[t]['data'],
                                agg_data=sub_data,
                                cat_var=c,
                                figsize=(20, 20),
                                ylabel=f'{t}_{cat_var}_{p_cat}',
                                pdf=pdf,
                                ylim=tags[t]['ylim'],
                                )
    return ppc_data


def scada_data_clustering(
    scada_tags,
    agg_data,
    sort_var,
    targets,
    cat_vars=[],
    agg_tags=None,
    max_clusters=6,
    min_clusters=2,
    rmv_out=3,
    prefix='',
    suffix='',
    save_pmml=False,
    build_pipe_func=std_build_pipe,
    show_metrics=True,
    full=True,
    print_profiles = True,
    profiles_by_cat = True
):
    """
    Function that produces two files:
    - file doing the clusteirng, and showing the target variables distribution per cluster
    - file visualising the scada time series grouped by cluster

    scada_tags: dict
        example provided at the end of this file
    agg_data: dataframe
        aggregated data
    path: str
        path where we want to store the files generated
    prefix: str
        name of hte prefix to use when naming the models and the files
    suffix: str
        suffix to append at the end of hte file name. can be used to version files created. i.e. 'v_0'
    DL: DataLoader instance
    sort_var: str
        DateTime variable to sort data
    targets: str or list
        used as targets in the analysis
    cat_vars: str or list
        i.e. plasma cluster (categorical variable)
    max_clusters: int
    min_clusters: int
    debug=False
        set tpo True if you want to do some prints when it's failing
    """

    old_imp = True
    try:
        ds = scada_tags[list(scada_tags.keys())[0]]['descriptive_statistics']
        print('Using old implementation with descriptive statistics')
    except:
        old_imp = False
        var_names = []

    if old_imp:
        data_eng = agg_data.drop([i for i in agg_data.columns if prefix in i], axis=1).copy()
    else:
        data_eng = agg_data.copy()

    for t in scada_tags.keys():

        if old_imp:
            if scada_tags[t]['descriptive_statistics']['mean']:
                data_eng = data_eng.merge(
                    right=scada_tags[t]['data_res'].mean(1).rename(prefix + '_' + t + '_mean'),
                    left_index=True,
                    right_index=True,
                    how='left',
                )
            if scada_tags[t]['descriptive_statistics']['std']:
                data_eng = data_eng.merge(
                    right=scada_tags[t]['data_res'].std(1).rename(prefix + '_' + t + '_std'),
                    left_index=True,
                    right_index=True,
                    how='left',
                )
            if scada_tags[t]['descriptive_statistics']['Q1']:
                data_eng = data_eng.merge(
                    right=scada_tags[t]['data_res']
                    .quantile(0.25, axis=1)
                    .rename(prefix + '_' + t + '_Q1'),
                    left_index=True,
                    right_index=True,
                    how='left',
                )
            if scada_tags[t]['descriptive_statistics']['Q3']:
                data_eng = data_eng.merge(
                    right=scada_tags[t]['data_res']
                    .quantile(0.75, axis=1)
                    .rename(prefix + '_' + t + '_Q3'),
                    left_index=True,
                    right_index=True,
                    how='left',
                )
            if scada_tags[t]['descriptive_statistics']['duration']:
                data_eng = data_eng.merge(
                    right=scada_tags[t]['data_res']
                    .count(axis=1)
                    .rename(prefix + '_' + t + '_duration'),
                    left_index=True,
                    right_index=True,
                    how='left',
                )
        else:
            if scada_tags[t]['info']['to_cluster']:
                if scada_tags[t]['info']['resample'] is not None:
                    dt = scada_tags[t]['data_res'].copy()
                else:
                    dt = scada_tags[t]['data'].copy()
                #dt.columns = [f'{t}_{int(i.total_seconds())}' for i in dt.columns]
                var_names += list(dt.columns)
                data_eng = data_eng.merge(right=dt, left_index=True, right_index=True, how='inner')

    folder_name = f'{prefix}_clustering_{suffix}'

    if old_imp:
        var_names = [i for i in data_eng.columns if (prefix in i)]
        data_eng = data_eng.dropna(subset=var_names)

    if agg_tags is not None:
        var_names += agg_tags
        var_detail = agg_tags
    else:
        var_detail = None
    

    ppc_data = cluster_params(
        var_list=var_names,
        data=data_eng,
        path=folder_name,
        targets=targets,
        sort_var=sort_var,
        cat_vars=cat_vars,
        var_detail=var_detail,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        rmv_out=rmv_out,
        prefix=prefix,
        suffix=suffix,
        show_metrics=show_metrics,
        save_pmml=save_pmml,
        build_pipe_func=build_pipe_func,
        to_pdf=full,
        time_series=False
    )

    for i in range(min_clusters, max_clusters + 1):
        ppc_data[f'{prefix}_{i}_clusters'] = ppc_data[f'{folder_name}_{i}_clusters'].astype(float)

    
    if print_profiles:
        with PdfPages(os.path.join(folder_name + '_models', f'{folder_name}_timeseries.pdf')) as pdf:

            title = t
            if scada_tags[t]['info']['step'] is not None:
                title = scada_tags[t]['info']['step'] + ' - ' + scada_tags[t]['info']['role'] + ' - ' + scada_tags[t]['info']['tag']

            for t in scada_tags.keys():
                if scada_tags[t]['info']['to_cluster'] == True:
                    plot_scada_heatmap_with_left_bar(
                        tag_data=scada_tags[t]['data_res'],
                        agg_data=ppc_data,
                        sort_batch_by=[sort_var],
                        sort_var=sort_var,
                        left_bar_CQA=targets,
                        title= title,
                        pdf=pdf,
                        vmax=None,
                        vmin=None,
                        val_cnt=False,
                        max_length=scada_tags[t]['graph']['max_length'],
                        start=scada_tags[t]['info']['start'],
                        end=scada_tags[t]['info']['end'],
                    )

            for i in range(min_clusters, max_clusters + 1):
                for t in scada_tags.keys():

                    title = t
                    if scada_tags[t]['info']['step'] is not None:
                        title = scada_tags[t]['info']['step'] + ' - ' + scada_tags[t]['info']['role'] + ' - ' + scada_tags[t]['info']['tag']

                    c = f'{prefix}_{i}_clusters'
                    plot_scada_heatmap_with_left_bar(
                        tag_data=scada_tags[t]['data_res'],
                        agg_data=ppc_data,
                        sort_batch_by=[c],
                        sort_var=sort_var,
                        left_bar_CQA=targets + [c],
                        title= title,
                        pdf=pdf,
                        vmax=None,
                        vmin=None,
                        val_cnt=False,
                        max_length=scada_tags[t]['graph']['max_length'],
                        start=scada_tags[t]['info']['start'],
                        end=scada_tags[t]['info']['end'],
                    )

                    cs = ppc_data[c].sort_values().unique()
                    mean_profile = plot_profile(
                        tag_data=scada_tags[t]['data_res'],
                        agg_data=ppc_data,
                        cat_var=c,
                        figsize=(20, 20),
                        ylabel=t,
                        pdf=pdf,
                        ylim=scada_tags[t]['graph']['ylim'],
                        max_length=scada_tags[t]['graph']['max_length'],
                        return_mean_profile=cs,
                    )

                    plot_dataframe(mean_profile.groupby(c).mean_profile.describe(), pdf_obj=pdf)

                    if profiles_by_cat:
                        
                        for cat_var in cat_vars:
                            for p_cat in ppc_data[cat_var].unique():
                                sub_data = ppc_data[ppc_data[cat_var] == p_cat]

                                plot_profile(
                                tag_data=scada_tags[t]['data_res'],
                                agg_data=sub_data,
                                cat_var=c,
                                figsize=(20, 20),
                                ylabel=f'{t}_{cat_var}_{p_cat}',
                                pdf=pdf,
                                ylim=scada_tags[t]['graph']['ylim'],
                                max_length=scada_tags[t]['graph']['max_length'],
                                )
    return ppc_data
