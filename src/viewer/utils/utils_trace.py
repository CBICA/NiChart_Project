# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def scatter_trace(df, xvar, yvar, hvar, fig):
    dft = df.copy()
    if hvar is None:
        hvar = 'All'
        dft['All'] = 1
    for hname, dfh in dft.groupby(hvar):
        trace = go.Scatter(
            x=dfh[xvar],
            y=dfh[yvar],
            mode='markers',
            name=hname,
            legendgroup=hname,
        )
        fig.add_trace(trace)

def linreg_trace(df: pd.DataFrame, xvar: str, yvar: str, hvar: str, fig: Any) -> Any:

    for hname, dfh in df.groupby(hvar):
        model = LinearRegression().fit(
            np.array(dfh[xvar]).reshape(-1, 1), (np.array(dfh[yvar]))
        )
        y_hat = model.predict(np.array(df[xvar]).reshape(-1, 1))
        trace = go.Scatter(
            x=df[xvar],
            y=y_hat,
            # showlegend=False,
            mode="lines",
            name=f'lin_{hname}',
            legendgroup=hname,
        )
        fig.add_trace(trace)  # plot in first row
    return fig


def lowess_trace(df: pd.DataFrame, xvar: str, yvar: str, fig: Any) -> Any:
    lowess = sm.nonparametric.lowess

    # y_hat = lowess(np.array(df[yvar], np.array(df[xvar], frac=1./3)
    y_hat = lowess(df[yvar], df[xvar], frac=1.0 / 3)
    trace = go.Scatter(
        x=y_hat[:, 0],
        y=y_hat[:, 1],
        showlegend=False,
        mode="lines",
        name="lowessfit",
        line=dict(color="rgb(0,255,0)"),
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig
    


def selid_trace(df: pd.DataFrame, sel_mrid: str, xvar: str, yvar: str, fig: Any) -> Any:
    df_tmp = df[df.MRID == sel_mrid]
    fig.add_trace(
        go.Scatter(
            x=df_tmp[xvar],
            y=df_tmp[yvar],
            mode='markers',
            name='Selected',
            marker=dict(color='rgba(250, 50, 50, 0.5)',
                        size=12,
                        line=dict(color='Red', width=3)
                        )
            )
    )


def percentile_trace(df: pd.DataFrame, xvar: str, yvar: str, fig: Any) -> Any:

    cline = [
        "rgba(255, 255, 255, 0.8)",
        "rgba(255, 225, 225, 0.8)",
        "rgba(255, 187, 187, 0.8)",
        "rgba(255, 0, 0, 0.8)",
        "rgba(255, 187, 187, 0.8)",
        "rgba(255, 225, 225, 0.8)",
        "rgba(255, 255, 255, 0.8)",
    ]

    cfan = [
        "rgba(255, 225, 225, 0.3)",
        "rgba(255, 225, 225, 0.3)",
        "rgba(255, 187, 187, 0.3)",
        "rgba(255, 0, 0, 0.3)",
        "rgba(255, 0, 0, 0.3)",
        "rgba(255, 187, 187, 0.3)",
        "rgba(255, 225, 225, 0.3)",
    ]

    df_tmp = df[df.ROI == yvar]

    # Create line traces
    for i, cvar in enumerate(df.columns[2:]):
        if i == 0:
            ctrace = go.Scatter(
                x=df_tmp[xvar],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                line=dict(color=cline[i]),
            )
        else:
            ctrace = go.Scatter(
                x=df_tmp[xvar],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                line=dict(color=cline[i]),
                fill="tonexty",
                fillcolor=cfan[i],
            )

        fig.append_trace(ctrace, 1, 1)  # plot in first row

    return fig


def dots_trace(df: pd.DataFrame, xvar: str, yvar: str) -> Any:
    trace = go.Scatter(
        x=df[xvar],
        y=df[yvar],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace


