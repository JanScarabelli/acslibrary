from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from acslibrary.plots.tables import plot_dataframe


def compute_regression_analysis(data, target: str, year_bias=False, categorical_bias=[]):
    """
    Compute the regression analysis using statsmodels.api for a given target.
    It includes dummy variables if needed to take into account potential biases in the data.
    Arguments:
    ------------
    data: dataframe
    target: str
        The target to predict with the regression analysis
    year_bias: str
        Datetime (or string) variable to use to compute year categories
    categorical_bias: list of strings
        all categorical variables to include in the analysis. Dummies will be created for
        each of these variables

    Return:
    ----------
    summary of the test that can be plotted

    """
    if year_bias != False:
        data[year_bias] = pd.to_datetime(data[year_bias])
        data['year'] = data[year_bias].apply(lambda x: x.year)
        for i in data['year'].value_counts().keys():
            data[i] = data['year'].apply(lambda x: 1 if x == i else 0)
    else:
        None
    cat_biases_include_analysis = []
    if len(categorical_bias) > 0:
        for var in categorical_bias:
            var_values = list(data[var].value_counts().index)
            for value in var_values[:-1]:
                data[var + '-' + str(value)] = data[var].apply(lambda x: 1 if x == value else 0)
                cat_biases_include_analysis.append(var + '-' + str(value))

    var_OLS = cat_biases_include_analysis
    var_OLS = (
        var_OLS + list(data['year'].value_counts().keys()[:-1]) if year_bias != False else var_OLS
    )
    d = data[var_OLS + [target]]
    x = d[var_OLS]
    X = sm.add_constant(x)
    y = d[[target]]

    model = sm.OLS(y, X)
    results = model.fit()
    return results.summary2()


def plot_regression_analysis(data, target: str, year_bias=False, categorical_bias=[], pdf=None):
    results = compute_regression_analysis(
        data=data, target=target, year_bias=year_bias, categorical_bias=categorical_bias
    )
    plt.figure(figsize=(15, 15))
    plt.text(0.1, 0.1, results.as_text(), fontfamily='monospace')
    if pdf != None:
        pdf.savefig()
    else:
        plt.show()
