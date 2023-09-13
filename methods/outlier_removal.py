import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # print(((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).sum())

    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)].dropna().copy()


def IQR_topN(df, n):
    mask = df.apply(lambda x: remove_outliers(x, n))
    return df[~mask.any(axis=1)].dropna().copy()


def remove_outliers(x, n):
    q1 = x.quantile(0.25)
    q2 = x.quantile(0.5)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    top5 = (abs(q2 - x)).sort_values(ascending=False)[:n].index
    return ((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))) & (x.index.isin(top5))


def detect_outliers(df, col_name, method, q_low=None, q_high=None):
    """
    Detects outliers using two different methods: 'IRQ' and 'parcentile capping'. Then it plots them on a scatter plot of the whole dataset and also 
    plots a histogram of the dataset with red vertical lines indicating where the distribution is cut (determining what are and aren't outliers).
    It returns the datframe without outliers and the batch IDs of the outliers removed.
    
    Parameters:
    ------------
    df: pandas dataframe
        dataframe whith all the dataset that we are using
    col_name: str
        name of the column in the dataframe df, where we want to detect outliers
    method: str
        name of the method of outlier detection that we want to use: 'IRQ' or 'parcentile capping'
    q_low: float
        if the method is 'parcentile capping', we can specify the lower quantile we want to cut at (optional)
    q_high: float
        if the method is 'parcentile capping', we can specify the higher quantile we want to cut at (optional)
    """
    
    if method == 'IQR':
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3-q1 # interquartile range
        
        q_low = q1-1.5*iqr
        q_high = q3+1.5*iqr
        
        df_out = df.loc[(df[col_name] > q_low) & (df[col_name] < q_high)]
        outliers = df.loc[~df.index.isin(df_out.index)]
        
    elif method == 'Percentile capping':
        if q_low != None:
            q_low = df[col_name].quantile(q_low)
        else: 
            q_low = df[col_name].min() - 1
            print("No lower parcentile capping applied")
            
        if q_high != None:
            q_high = df[col_name].quantile(q_high)
        else: 
            q_high = df[col_name].max() + 1
            print("No higher parcentile capping applied")
            
        df_out = df.loc[(df[col_name] > q_low) & (df[col_name] < q_high)]
        outliers = df.loc[~df.index.isin(df_out.index)]
        
    else:
        print("Use one of the defined methods: IQR, Percentile capping")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # scatter plot to see the outliers detected
    ax1.scatter(df[col_name],df.index,marker='.',label='original data')
    ax1.scatter(outliers[col_name],outliers.index, s=80, facecolors='none', edgecolors='r', label='outliers')
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Batches')
    ax1.set_yticks([])
    ax1.legend()
  
    # plot histogram indicating where the distribution was cut
    ax2.hist(df[col_name])
    ax2.axvline(x=q_low, lw=2, ls='--', color='r')
    ax2.axvline(x=q_high, lw=2, ls='--', color='r')
    ax2.set_xlabel(col_name)
    ax2.set_ylabel('Frequency')
    
    return df_out, outliers.index
