import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from statsmodels.formula.api import ols

from acslibrary.plots.tables import plot_dataframe
from acslibrary.plots.boxplots import draw_boxplots
from acslibrary.methods.posthoc_tests import tukey


def ANOVA(
    data, target, cat1, cat2=None, interaction=True, typ=2, pval=0.05, do_tukey=False, pdf_obj=None
):
    """
    Completes ANOVA test on 1 or 2 variables (also known as One-Way or Two-Way): cat1 and cat2 
    (optional) against a target. Tukey test is optional. You can also adjuste the p-value in 
    order to accept the Null Hypostesis
    There is the possibility to save results in a pdf file.

    Parameters:
    -----------
    data: dataframe
        dataframe that must contain cat1, cat2 and target. rows containing NaNs will be removed
    target: str
        target variable
    cat1, cat2: str, cat2 is optional. 
        You have to put the name of the variables in the column that will be used as categories.
    interaction: bool
        Applies (or not) interaction between the two categories
    typ: int
        Allows you to choose the type of Anova Test. 
        For more information: https://www.r-bloggers.com/2011/03/anova-%e2%80%93-type-iiiiii-ss-explained/
    pval: float
        P-value to consider a statistical significance
    tukey: bool, optional
        Whether to compute or not the tukey test.
        For more information: https://en.wikipedia.org/wiki/Tukey%27s_range_test
    pdf: None or a matplotlib.backends.backend_pdf.PdfPages instance
        pdf to output results

    Return:
    --------
    Nothing really, prints a lot of things or fill a pdf file
    """
    df = data[[target, cat1, cat2]].dropna() if cat2 != None else data[[target, cat1]].dropna()
    if cat2 != None:
        if interaction != True:
            # Two way ANOVA without interaction
            ols_statement = target + ' ~ ' + 'C(' + cat1 + ')' + ' + ' + 'C(' + cat2 + ')'
        else:
            # Two way ANOVA with interaction
            ols_statement = (
                target
                + ' ~ '
                + 'C('
                + cat1
                + ')'
                + ' + '
                + 'C('
                + cat2
                + ')'
                + ' + C('
                + cat1
                + '):C('
                + cat2
                + ')'
            )
        # Plots
        # f, (a0, a1) = plt.subplots(1, 2, figsize=(15,7))
        # sns.boxplot(x=cat1, y=target, hue=cat2, data=df, ax=a0, showmeans=True) #, palette="Set1"
        # sns.boxplot(x=cat2, y=target, hue=cat1, data=df, ax=a1, showmeans=True) #, palette="Set1"
        # plt.show() if pdf_obj==None else pdf_obj.savefig()
        descr_df = df.groupby(by=[cat1, cat2])[[target]].describe()
        print(descr_df) if pdf_obj == None else plot_dataframe(
            descr_df, pdf_obj, ttle=f'{cat1} and {cat2} description'
        )

    else:
        # One way ANOVA
        ols_statement = target + ' ~ ' + 'C(' + cat1 + ')'
        # Plots
        draw_boxplots(
            x=cat1,
            Y=[target],
            data=df,
            order=np.sort(df[cat1].unique()),
            category=None,
            pal='Set2',
            pdf_obj=pdf_obj,
            fig_size=(15, 7),
        )  # , palette="Set1"

        descr_df = df.groupby(by=cat1)[[target]].describe().reset_index()
        descr_df.columns = descr_df.columns.droplevel()

        # fig, ax = plt.subplots(2, 1, figsize=(10,4))
        fig = plt.figure(figsize=(10, 4))
        outer = gridspec.GridSpec(2, 1)
        plot_dataframe(descr_df, outer_spec=outer[0], fig=fig, ttle=cat1 + ' description')

    if pdf_obj == None:
        print('')
        print('')
        print('**** VARIABLE SIGNIFICANCE ****')
        print('')
        print('OLS Statement: ' + ols_statement)
        print('')

        # Ordinary Least Squares (OLS) model
        model = ols(ols_statement, data=df).fit()
        avt = sm.stats.anova_lm(model, typ=typ)

        print(avt)
        print('')
        print('samples per category cat1: \n', df[cat1].value_counts())
        for indx, rows in avt.iterrows():
            if indx != 'Residual':
                if rows['PR(>F)'] < pval:
                    print(indx + ' - is significant with a pvalue of ' + str(rows['PR(>F)']))
                else:
                    print(
                        indx
                        + ' - is NOT significant, not enough evidence to reject null hypothesis'
                    )
    else:
        model = ols(ols_statement, data=df).fit()
        avt = sm.stats.anova_lm(model, typ=typ)
        fig = plt.figure(figsize=(10, 4))
        outer = gridspec.GridSpec(2, 1)
        plot_dataframe(avt, pdf_obj=pdf_obj, outer_spec=outer[1], fig=fig, ttle='ANOVA Results')
    if do_tukey:
        # Tukey pairwise tests
        analysis_subset = df.dropna(how='any', subset=[target])
        tuk_all, tuk_rejected = tukey(analysis_subset, target, cat1)
        if pdf_obj == None:
            print('')
            print('')
            print('**** PAIRWISE ANALYSIS ****')
            print('')
            print(cat1)
            print('')
            print(tuk_all.sort_values('p-adj').head(40))
            print('')
            print(tuk_rejected.sort_values('total_sum', ascending=False)[['total_sum']].head(40))
            print(
                '*Note: "total_sum" represents the number of times the null hypothesis was rejected for the value on the left'
            )
        else:
            plot_dataframe(tuk_all.sort_values('p-adj').head(40), pdf_obj, ttle=cat1)
        if cat2 != None:
            tuk_all, tuk_rejected = tukey(analysis_subset, target, cat2)
            if pdf_obj == None:
                print('')
                print('')
                print(cat2)
                print('')
                print(tuk_all.sort_values('p-adj').head(40))
                print('')
                print(
                    tuk_rejected.sort_values('total_sum', ascending=False)[['total_sum']].head(40)
                )
                print(
                    '*Note: "total_sum" represents the number of times the null hypothesis was rejected for the value on the left'
                )
            else:
                plot_dataframe(tuk_all.sort_values('p-adj').head(40), pdf_obj, ttle=cat2)
            if interaction != False:
                var_combo = cat1 + ' | ' + cat2
                analysis_subset.loc[:, 'combined'] = (
                    analysis_subset.loc[:, cat1].astype(str)
                    + ' | '
                    + analysis_subset.loc[:, cat2].astype(str)
                )
                # analysis_subset.loc[:,'combined'] = analysis_subset.loc[:,cat1].astype(str) + ' | '
                tuk_all, tuk_rejected = tukey(analysis_subset, target, 'combined')
                if pdf_obj == None:
                    print('')
                    print('')
                    print(var_combo)
                    print('')
                    print(tuk_all.sort_values('p-adj').head(40))
                    print('')
                    print(
                        tuk_rejected.sort_values('total_sum', ascending=False)[['total_sum']].head(
                            40
                        )
                    )
                    print(
                        '*Note: "total_sum" represents the number of times the null hypothesis \
                    was rejected for the value on the left'
                    )
                else:
                    plot_dataframe(
                        tuk_all.sort_values('p-adj').head(40),
                        pdf_obj,
                        ttle=var_combo,
                        fig_size=(10, 15),
                    )
