import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def tukey(df, target, group, pval=0.05):
    """
    Computes the tukey test
    """

    m_comp = pairwise_tukeyhsd(endog=df[target], groups=df[group], alpha=pval)
    tukey_data = pd.DataFrame(
        data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0]
    )

    group1_comp = tukey_data.loc[tukey_data.reject == True].groupby('group1').reject.count()
    group2_comp = tukey_data.loc[tukey_data.reject == True].groupby('group2').reject.count()
    tukey_data_sub = pd.concat([group1_comp, group2_comp], axis=1)

    tukey_data_sub = tukey_data_sub.fillna(0)
    tukey_data_sub.columns = ['reject1', 'reject2']
    tukey_data_sub.loc[:, 'total_sum'] = tukey_data_sub.reject1 + tukey_data_sub.reject2

    tukey_data_sub.sort_values('total_sum', ascending=False).head(20)

    return tukey_data, tukey_data_sub
