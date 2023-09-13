import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
from acslibrary.methods.posthoc_tests import tukey


def intercalated_lists(list1, list2):
    newlist = []
    a1 = len(list1)
    a2 = len(list2)

    for i in range(max(a1, a2)):
        if i < a1:
            newlist.append(list1[i])
        if i < a2:
            newlist.append(list2[i])

    return newlist


def test_pairs(data, category, y, order, ret_tests=True):

    ret = pd.DataFrame()
    tested = True

    try:
        obj1, obj2 = tukey(data.dropna(), y, category)
    except:
        tested = False
        obj1 = pd.DataFrame()

    counts = data.groupby(category)[y].count()
    l = len(order)
    i = 0
    while i <= l - 1:

        if i == l - 1:
            inx0 = 0
            inx1 = l - 1
        else:
            inx0 = i
            inx1 = i + 1

        if tested:
            res = obj1[(obj1.group1 == order[inx0]) & (obj1.group2 == order[inx1])]['p-adj']
            if len(res) == 0:
                res = obj1[(obj1.group2 == order[inx0]) & (obj1.group1 == order[inx1])]['p-adj']
            try:
                p = res.values[0]
            except:
                p = np.nan
        else:
            p = np.nan

        try:
            n1 = counts[order[inx0]]
        except:
            n1 = 0

        try:
            n2 = counts[order[inx1]]
        except:
            n2 = 0

        ret[order[i]] = [str(round(p, 2)), str(n1), str(n2)]
        i += 1

    if ret_tests:
        return ret, obj1
    else:
        return ret


def ttest_format(t, grouped=True):

    if grouped:
        t = t.reset_index()
        return ' (p = ' + t.iloc[0, 2] + ')', 'nobs = ' + t.iloc[1, 2] + ', ' + t.iloc[2, 2]
    else:
        return (' (p = ' + t[0] + ')', 'nobs = ' + t[1], 'nobs = ' + t[2])


def draw_boxplots(
    data,
    Y,
    x,
    category,
    order=None,
    info=None,
    split_x=False,
    split_x_labels=[],
    fig_size=(20, 5),
    rotate_x=0,
    p_vals_only=False,
    pdf_obj=None,
    pal='Blues',
    title='',
    subtitle='',
    scale = 1,
):

    r"""Prints a (grid of) boxplot(s), showing tbe number of observations contained
    in each category and p-values associated with t-tests for the difference in means
    between combinations of categories.

    Three types of grid are available:
        1. One y axis variable per cell (`Y`), category in x axis (`x`)
        2. One y axis variable per cell (`Y`), main category on x axis (`x`),
           further divided by a second category (`category`)
        3. Same variable per row in y axis (`Y`), category in x axis (`category`),
           x axis further separated by subgroups (`x`)

    The tests for the difference in means are always performed pairwise in the order
    indicated by the user (`order`) or in the order they appear in `data`. It always
    includes a test between the first and last category. For instance, if
    `order = ['A','B','C','D']` (or this is the order the categories are found in `data`),
    p-values are shown for A vs B, B vs C, C vs D, and A vs D.

    For a type 2 grid, `category` can only contain 2 different categories and
    p-values are shown only for these categories, and not the ones expressed by `x`.

    Parameters
    ----------
    data : 2d array_like
        Observations to be plotted
    Y : list
        Strings with the names of the variables to be plotted in the y axis
    x : str
        The name of the category to be represented by the boxes on a type 1 grid, or the
        main category (to be further divided by `category`) on a type 2 grid, or the category
        to split the x axis in subgroups on a type 3 grid.
    category : str
        The name of the category to be represented by the boxes on a type 2 or type 3 grid.
        For a type 2 grid, this variable must necessarily contain only 2 categories. Must be
        set to `None` if the desired grid is a type 1 grid.
    order : list, optional
        Names (strings) of categories represented by the boxes. The boxes are ordered according
        to this variable. If `None`, boxes are ordered according to occurence in `data`.
    info: 1d pandas.DataFrame
        Description of `Y` variables, where indexes are `Y` variables and values are strings.
    split_x : bool, optional
        Must be set to `True` if the desired grid is a type 3 grid.
    split_x_labels : list, optional
        Strings. Changes the labels of `x` if `split_x = True`


    Returns
    -------
    tests : dict
        Numbers related to the t-tests, indexed by variables in `Y`

    Other Parameters
    ----------------
    fig_size : tuple, optional
        Width and length of the grid
    rotate_x : int, optional
        Degrees to rotate x axis labels
    p_vals_only : bool, optional
        If set to `True`, x axis labels are suppressed and only p-values are shown.
    to_pdf : bool, optional
        If set to `True`, matplotlib.pylab.show() is not called at the end.
    pal : str, optional
        Color pallete. Accepts every pallete available through the seaborn library.
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    scale : float, optional
        Plot scale factor for text size

    See Also
    --------
    ttest_pairs, intercalated_lists

    Examples
    --------
    Send request to talitha.speranza@aizon.ai

    """

    fig, ax = None, None
    ret = {}
    sns.set_style('white')
    sns.set(font_scale=scale)


    if split_x:
        X = np.sort(data[x].unique())
        ix = np.arange(len(X))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(Y), figsize=fig_size, sharex=False)
        plt.suptitle(title + '\n' + subtitle)
        ix = [0]

    if order is None and category is not None:
        order = np.sort(data[category].unique())
    elif order is None:
        order = np.sort(data[x].unique())

    iy = np.arange(len(Y))

    for i in iy:

        g = i

        if split_x:
            fig, ax = plt.subplots(
                nrows=1, ncols=len(X), figsize=fig_size, sharex=False, sharey=True
            )
            if i == 0:
                plt.suptitle(title + '\n' + subtitle)

        if len(Y) == 1:
            axg = ax
        else:
            axg = ax[g]

        for j in ix:

            if split_x:
                g = j
                axg = ax[g]
                wrkdata = data.loc[data[x] == X[j], :]
                sns.boxplot(
                    data=wrkdata,
                    x=category,
                    order=order,
                    y=Y[i],
                    ax=axg,
                    palette=pal,
                    showmeans=True,
                )
                axg.set_xlabel(x + ' ' + str(j))
            else:
                sns.boxplot(
                    data=data,
                    x=x,
                    y=Y[i],
                    order=order,
                    hue=category,
                    ax=axg,
                    palette=pal,
                    showmeans=True,
                )
                data[x] = pd.Categorical(data[x], categories=order)
                axg.set_xlabel('')
                wrkdata = data

            if split_x and j > 0:
                axg.set_ylabel('')
            else:
                try:
                    axg.set_ylabel(str(Y[i]) + ': ' + info.loc[Y[i], 'description'])
                except:
                    pass

            xtop = axg.secondary_xaxis('top')
            xtop.set_xticks(list(range(0, len(order))))
            # xtop.set_xticks(order)

            if category != None and not split_x:
                cats = np.sort(data[category].unique())
                summary = (
                    data[[x, Y[i], category]]
                    .groupby(x)
                    .apply(lambda g: test_pairs(g, category, Y[i], cats, ret_tests=False))
                    .reset_index()
                    .drop('level_1', axis=1)
                )
                tests = summary
                formatted = summary.groupby(x).apply(lambda t: ttest_format(t.iloc[:3, :]))
                formatted = pd.DataFrame(list(formatted))
                axg.set_xticklabels(
                    [str(k) + str(l) for k, l in zip(order, formatted.iloc[:, 0])],
                    rotation=rotate_x,
                )
                xtop.set_xticklabels(list(formatted.iloc[:, 1]), rotation=rotate_x)

                if i != len(Y) - 1:
                    axg.get_legend().set_visible(False)
            else:
                if split_x:
                    summary, tests = test_pairs(wrkdata[[category, Y[i]]], category, Y[i], order)
                    axg.set_xlabel(X[j])
                else:
                    summary, tests = test_pairs(wrkdata[[x, Y[i]]], x, Y[i], order)

                formatted = summary.apply(lambda t: ttest_format(t, grouped=False))
                xtop.set_xticklabels(
                    [formatted.iloc[1, 0]] + list(formatted.iloc[2, :-1]), rotation=rotate_x
                )
                axg.set_xticks(np.arange(0, formatted.shape[1] - 0.5, 0.5))

                if p_vals_only:
                    labs = list(np.repeat('', len(order)))
                else:
                    labs = order

                axg.set_xticklabels(
                    intercalated_lists(labs, list(formatted.iloc[0, :-1])), rotation=rotate_x
                )

                if formatted.shape[1] > 2:
                    p = str(order[0]) + '/' + str(order[-1]) + ': ' + formatted.iloc[0, -1]
                    axg.legend([p], loc='best', fontsize='medium', handlelength=0, handletextpad=0)

        ret[Y[i]] = tests

    if rotate_x == 0:
        fig.text(0.5, 0.005, x, ha='center')
    else:
        fig.text(0.5, -0.06, x, ha='center')

    if pdf_obj != None:
        pdf_obj.savefig() # change
    else:
        plt.show()
    plt.close('all')
    return ret, ax
