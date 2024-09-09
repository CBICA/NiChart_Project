# -*- coding: utf-8 -*-
import math
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from plotly import tools
import statsmodels.api as sm
import numpy as np

####### Plot types ######

def percentile_trace(df, xvar, yvar, fig):
    
    cline = ['rgba(255, 255, 255, 0.8)', 'rgba(255, 225, 225, 0.8)', 
             'rgba(255, 187, 187, 0.8)', 'rgba(255, 0, 0, 0.8)',
             'rgba(255, 187, 187, 0.8)', 'rgba(255, 225, 225, 0.8)', 'rgba(255, 255, 255, 0.8)']

    cfan = ['rgba(255, 225, 225, 0.3)', 'rgba(255, 225, 225, 0.3)', 
            'rgba(255, 187, 187, 0.3)', 'rgba(255, 0, 0, 0.3)',
            'rgba(255, 0, 0, 0.3)', 'rgba(255, 187, 187, 0.3)', 'rgba(255, 225, 225, 0.3)']
    
    df_tmp = df[df.ROI == yvar]
        
    # Create line traces
    for i,cvar in enumerate(df.columns[2:]):
        if i == 0:
            ctrace = go.Scatter(x = df_tmp[xvar], y = df_tmp[cvar], 
                                mode='lines', name = cvar,
                                line = dict(color = cline[i]))
        else:
            ctrace = go.Scatter(x = df_tmp[xvar], y = df_tmp[cvar], 
                                mode='lines', name = cvar, 
                                line = dict(color = cline[i]),
                                fill = 'tonexty',
                                fillcolor = cfan[i])

        fig.append_trace(ctrace, 1, 1)  # plot in first row

    return fig

def dots_trace(df, xvar, yvar):
    trace = go.Scatter(
        x=df[xvar], y=df[yvar], showlegend=False, mode = 'markers', name = "datapoint",
        line = dict(color = 'rgb(0,160,250)'),
    )
    return trace

def linreg_trace(df, xvar, yvar, fig):
    model = LinearRegression().fit(np.array(df[xvar]).reshape(-1,1), 
                                   (np.array(df[yvar])))
    y_hat = model.predict(np.array(df[xvar]).reshape(-1,1))
    trace = go.Scatter(
        x=df[xvar], y=y_hat, showlegend=False, mode = 'lines', name = "linregfit",
        line = dict(color = 'rgb(0,0,255)'),
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig

def lowess_trace(df, xvar, yvar, fig):
    lowess = sm.nonparametric.lowess

    #y_hat = lowess(np.array(df[yvar], np.array(df[xvar], frac=1./3)
    y_hat = lowess(df[yvar], df[xvar], frac=1./3)
    trace = go.Scatter(
        x = y_hat[:,0], y=y_hat[:,1], showlegend=False, mode = 'lines', name = "lowessfit",
        line = dict(color = 'rgb(0,255,0)'),        
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig
