# -*- coding: utf-8 -*-
"""
Created on July 1 19:56:28 2022

AUTHOR:     William Nadolski

NAME:       bsplines.py

PURPOSE:    Plot fitted bspline trend to data series

INPUTS:     pandas dataframe

OUTPUTS:    None

DEPENDENCY: None
    
PARAMS:     df, ycol, datecol, datefmt, nknots, xtickperiod, ptitle, xlabel, ylabel

EXCEPTIONS: None

LOGIC:      Script performs the following tasks:
            1. Generates syntehtic data frame for testing
            2. Computes min and max date in data, sorts by date
            3. Formats date, determines desired xtick frequency
            4. Fits bspline to data using specified number of internal knots
            5. Plots original series and overlaid bspline

NOTES:      Function assumes x-axis is daily date

TODO:       Change graph colors and font to match aizon styling
            Allow function to accept and iterate over list of input cols (Y)
            Generalize function to accept non-daily date values for x-axis
            
CHANGES:    WN - 04AUG2022 - Initial commit
    
"""

#%% IMPORT NECESSARY PACKAGES

import os
import datetime as dt
import seaborn as sns
from datetime import datetime

import json
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from itertools import cycle
from scipy.interpolate import BSpline, splrep, splev

#%% DEFINE FUNC TO GENERATE SYNTHETIC DATA ###

#generate syntehtic wave function for one year
def generate_synthetic_data(t=365, distinct_dates=True):

    x = np.arange(0,t)
    s =  np.sin(4*np.pi*x/t)+np.cos(8*np.pi*x/t)
    plt.title("Wave Signal")
    plt.plot(x, s)
    plt.show()
    
    #generate date index
    today = datetime.now().date()
    date = pd.date_range(today, periods=t, freq='D')
    
    #combine date and signal
    d = {"date": date, "signal": s}
    df = pd.DataFrame(d)
    
    #generate noise and add to signal
    df['noise'] = np.random.normal(-0.5, 1, [t,1]) 
    df['value'] = df['signal'] + df['noise']
    
    #generate another set of data to append to original
    #meant to represent situations where time index is not unique
    if distinct_dates == False:
        df2 = pd.DataFrame(d)
        df2['noise'] = np.random.normal(-0.5, 1, [t,1]) 
        df2['value'] = df2['signal'] + df2['noise']    
        df = df.append(df2)
        df = df.reset_index(drop=True)
    
    #plot resulting series
    plt.title("Noisy Series")
    plt.scatter(df['date'], df['value'], s=5)
    plt.show()
    
    return df

#%% DEFINE FUNC TO PLOT BSPLINE ###

def plot_bspline(
                 df = df, 
                 ycol = 'YVAR', 
                 datecol = 'DATE', 
                 datefmt = '%Y-%m-%d', 
                 nknots = 5, 
                 xtickperiod = 7, 
                 ptitle = "Fitted B-Spline", 
                 xlabel = "Date", 
                 ylabel = "Y"
                 ):

    df = df.copy()
    df = df[[datecol, ycol]]
    df = df.dropna(axis='index', subset=[ycol])
    mindate = df[datecol].min()
    maxdate = df[datecol].max()
    df['days'] = (df[datecol] - mindate).dt.days
    df = df.sort_values(by='days')
    df.reset_index(drop=True)
    df = df[~df.duplicated(subset='days', keep='first')]
    x = df['days']
    y = df[ycol]
    z = pd.to_datetime(df[datecol]).dt.strftime(datefmt)
    ticks = round(x.shape[0]/xtickperiod)  # number of x ticks to display
    # Fit
    qs = np.linspace(0, 1, nknots+2)[1:-1]
    knots = np.quantile(x, qs)
    tck = splrep(x, y, t=knots, k=3)
    ys_smooth = splev(x, tck)
    # Display
    fig, ax = plt.subplots(figsize=(15, 5))
    #plt.ylim([-4.0, 3.0])
    plt.plot(z, y, '.c')
    plt.plot(z, ys_smooth, '-m')
    plt.title(ptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=70)
    ax.xaxis.set_ticks(np.arange(0, x.shape[0], xtickperiod))
    plt.show()


#%% EXECUTE FUNCTIONS ###

df_synth = generate_synthetic_data(t=365, distinct_dates=False)

plot_bspline(
             df = df_synth, 
             ycol = 'value', 
             datecol = 'date', 
             datefmt = '%Y-%m-%d', 
             nknots = 7, #controls how much to over/underfit spline
             xtickperiod = 7, #using 7 to display one tick per week
             ptitle = "Fitted B-Spline", 
             xlabel = "Date", 
             ylabel = "Y"
             )