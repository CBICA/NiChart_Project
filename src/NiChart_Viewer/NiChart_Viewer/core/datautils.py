# This Python file uses the following encoding: utf-8
"""
contact: software@cbica.upenn.edu
Copyright (c) 2018 University of Pennsylvania. All rights reserved.
Use of this source code is governed by license located in license file: https://github.com/CBICA/NiChart_Viewer/blob/main/LICENSE
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import numpy as np
import joblib
import os, sys
from NiChart_Viewer.core import iStagingLogger
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMdiArea, QMdiSubWindow, QTextEdit, QComboBox, QLayout, QMessageBox, QErrorMessage

#from NiChart_Viewer.core.datautils import *

import statsmodels.formula.api as sm

logger = iStagingLogger.get_logger(__name__)

########################################################
## Plotting functions

def hue_regplot(data, x, y, hue, palette=None, **kwargs):
    '''Plotting
    '''
    
    data[x] = data[x].astype(float)
    data[y] = data[y].astype(float)
    
    regplots = []
    levels = data[hue].unique()
    if palette is None:
        default_colors = plt.colormaps['tab10']
        palette = {k: default_colors(i) for i, k in enumerate(levels)}
    legendhandls=[]
    for key in levels:
        regplots.append(sns.regplot(x=x, y=y, data=data[data[hue] == key], color=palette[key], **kwargs))
        legendhandls.append(Line2D([], [], color=palette[key], label=key))
    return (regplots, legendhandls)

def DataPlotDist(axes, df, x_var, hue_var):
    '''Plot
    '''

    # clear plot
    axes.clear()

    ## Plot distribution
    if len(hue_var)>0:
        sns.kdeplot(data=df, x=x_var, hue=hue_var, ax=axes)
    else:
        sns.kdeplot(data=df, x=x_var, ax=axes)
    sns.despine(fig=axes.get_figure(), trim=True)
    axes.get_figure().set_tight_layout(True)
    axes.set(xlabel=x_var)

def DataPlotScatter(axes, df, x_var, y_var, hue_var=''):
    '''Plot
    '''
    
    ## Get hue values
    if len(hue_var)>0:
        a,b = hue_regplot(df, x_var, y_var, hue_var, ax=axes)
        axes.legend(handles=b)
    else:
        sns.regplot(data = df, x = x_var, y = y_var, ax=axes)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    sns.despine(fig=axes.get_figure(), trim=True)
    axes.get_figure().set_tight_layout(True)
    axes.set(xlabel=x_var)
    axes.set(ylabel=y_var)
    
def DataPlotWithCentiles(axes, df, x_var, y_var, df_cent, cent_type, hue_var=''):
    '''Plot
    '''

    tmp_pal = ["#00ff00", "#0000ff", "#000000"]

    tmp_pal = ["#D2D68D", "#D8534D", "#12664F", "#2DC2BD", "#FB8B24", "#F5FF90"]
 
    tmp_pal = ["#00AF54", "#06070E", "#FB8B24", "#2A2B2A", "#706C61", "#FB8B24", "#E5446D"]

    hue_var_default = 'Sex'
    if hue_var not in df.columns:
        hue_var = hue_var_default
        
    hue_order = df[hue_var].sort_values().unique()
    
    sns.scatterplot(data = df, x = x_var, y = y_var, hue = hue_var, hue_order = hue_order,
                    palette = tmp_pal, ax=axes)
    
    df_tmp = df_cent[df_cent.ROI_Name == y_var]
    cent_vals = df_tmp.columns[df_tmp.columns.str.contains('centile')].tolist()
    df_tmp = pd.melt(df_tmp, id_vars=['Age'], value_vars = cent_vals)
    
    num_cent = len(cent_vals)
        
    tmp_pal = ['#f8baba', '#f87c7c', '#f83e3e', '#f80000',
               '#f83e3e', '#f87c7c', '#f8baba']
               
    g = sns.lineplot(data = df_tmp, x = x_var, y = 'value', hue = 'variable', 
                 palette = tmp_pal,
                 ax=axes)

    g.legend_.set_title(hue_var)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    sns.despine(fig=axes.get_figure(), trim=True)
    axes.get_figure().set_tight_layout(True)
    axes.set(xlabel=x_var)
    axes.set(ylabel=y_var)

    g.set_title('Centiles: ' + cent_type)


########################################################
## Data manipulation functions

def DataFilter(df, filter_var, filter_vals):
    '''Filter
    '''
    
    ## Get filter values
    if len(filter_var) == 0:
        out_code = 1
        out_msg = 'WARNING: Please select filter vars!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    is_numerical = pd.to_numeric(df[filter_var].dropna(), errors='coerce').notnull().all()

    if is_numerical:
        if len(filter_vals) != 2:
            out_code = 2
            out_msg = 'WARNING: Please select min / max values!'
            return {'out_code' : out_code, 'out_msg' : out_msg}
        df_out = df[ (df[filter_var] >= filter_vals[0]) & (df[filter_var] <= filter_vals[1])]
        
    else:
        if len(filter_vals) == 0:
            out_code = 3
            out_msg = 'WARNING: Please select filter values!'
            return {'out_code' : out_code, 'out_msg' : out_msg}
        df_out = df[df[filter_var].isin(filter_vals)]

    out_code = 0
    out_msg = 'Filtered data'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}

def DataSelectColumns(df, sel_cols):
    '''Select columns
    '''
    if len(sel_cols) == 0:
        out_code = 1
        out_msg = 'WARNING: Please select columns!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
        
    ## Select columns
    try:
        df_out = df[sel_cols]

    except:
        out_code = 2        
        out_msg = 'ERROR: Could not select columns!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    out_code = 0
    out_msg = 'Columns selected'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}

def DataGetStats(df, group_vars, display_vars, stat_vars):
    '''Stats
    '''
    ## Check validity of out vars and stats to display
    if len(display_vars) == 0:
        out_code = 1
        out_msg = 'WARNING: Please select input variable(s)!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    if len(stat_vars) == 0:
        out_code = 2
        out_msg = 'WARNING: Please select output stat(s)!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    df_out = df[group_vars + display_vars]
    
    if len(group_vars)>0:

        if df_out[group_vars].drop_duplicates().shape[0] > 20:
            out_code = 3
            out_msg = 'WARNING: Please check the group variables. Too many groups! (max allowed: 50)'
            return {'out_code' : out_code, 'out_msg' : out_msg}

        ## Get stats
        df_out = df_out.groupby(group_vars).describe()
        
        ## Select stats to display
        df_out = df_out.loc[:, pd.IndexSlice[:, stat_vars]]

        ## Change multiindex to single for display in table view
        df_out = df_out.reset_index()
        df_out = df_out.set_index(df_out.columns[0]).T
        df_out = df_out.reset_index(names = [group_vars[0], ''])

    else:
        ## Get stats
        df_out = df_out.describe()

        ## Select stats to display
        df_out = df_out.loc[stat_vars, :]

        ## Change multiindex to single for display in table view
        df_out = df_out.reset_index(names = 'Stats')

    out_code = 0
    out_msg = 'Created data stats'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}

def DataSort(df, sort_cols, sort_orders):
    '''Sort
    '''
    if len(sort_cols) == 0:
        out_code = 1        
        out_msg = 'WARNING: Please select sort column(s)!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    df_out = df.sort_values(sort_cols, ascending=sort_orders)
    
    out_code = 0
    out_msg = 'Created sorted data'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}
    
    
def DataMerge(df1, df2, mergeOn1, mergeOn2):
    '''Merge datasets
    '''

    if (len(mergeOn1) == 0) | (len(mergeOn2) == 0):
        out_code = 1        
        out_msg = 'WARNING: Please select merge column(s)!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    try:
        #df_out = df1.merge(df2, left_on = mergeOn1, right_on = mergeOn2, suffixes=['','_DUPLVARIND'])
        df_out = df2.merge(df1, left_on = mergeOn2, right_on = mergeOn1, suffixes=['','_DUPLVARIND'])
        
        ## If there are additional vars with the same name, we keep only the ones from the first dataset
        df_out = df_out[df_out.columns[df_out.columns.str.contains('_DUPLVARIND')==False]]

    except:
        out_code = 2        
        out_msg = 'ERROR: Could not merge dataframes!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    out_code = 0
    out_msg = 'Merged data tables'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}
    
    return df_out

def DataConcat(df1, df2):
    '''Merge datasets
    '''
    df_out = pd.concat([df1, df2])
    
    return df_out

def DataAdjCov(df, key_var, target_vars, cov_corr_vars, cov_keep_vars=[], 
               sel_col='', sel_vals = [], out_suff = 'COVADJ'):       
    '''Apply a linear regression model and correct for covariates
    It runs independently for each outcome variable
    The estimation is done on the selected subset and then applied to all samples
    The user can indicate covariates that will be corrected and not
    '''
    if key_var == '':
        out_code = 1        
        out_msg = 'WARNING: Please select primary key variable!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    # Combine covariates (to keep + to correct)
    if cov_keep_vars is []:
        covList = cov_corr_vars;
        isCorr = list(np.ones(len(cov_corr_vars)).astype(int))
    else:
        covList = cov_keep_vars + cov_corr_vars;
        isCorr = list(np.zeros(len(cov_keep_vars)).astype(int)) + list(np.ones(len(cov_corr_vars)).astype(int))

    # Prep data
    TH_MAX_NUM_CAT = 20     ## FIXME: This should be a global var
    df_covs = []
    isCorrArr = []
    for i, tmpVar in enumerate(covList):
        ## Detect if var is categorical
        is_num = pd.to_numeric(df[tmpVar].dropna(), errors='coerce').notnull().all()
        if df[tmpVar].unique().shape[0] < TH_MAX_NUM_CAT:
            is_num = False
        ## Create dummy vars for categorical data
        if is_num == False:
            dfDummy = pd.get_dummies(df[tmpVar], prefix=tmpVar, drop_first=True)
            df_covs.append(dfDummy)
            isCorrArr = isCorrArr + list(np.zeros(dfDummy.shape[1]).astype(int)+isCorr[i])
        else:
            df_covs.append(df[tmpVar])
            isCorrArr.append(isCorr[i])
    df_covs = pd.concat(df_covs, axis=1)
    
    ## Get cov names
    cov_vars = df_covs.columns.tolist()
    str_cov_vars = ' + '.join(cov_vars)
    
    ## Get data with all vars
    if sel_vals == []:
        df_train = df
    else:
        df_train = df[df[sel_col].isin(sel_vals)]
    
    ## Fit and apply model for each outcome var
    df_out = df[[key_var]].copy()
    for i, tmp_out_var in enumerate(target_vars):

        ## Fit model
        str_model = tmp_out_var + '  ~ ' + str_cov_vars
        mod = sm.ols(str_model, data = df_train)
        res = mod.fit()

        ## Apply model
        corrVal = df[tmp_out_var]
        for j, tmpCovVar in enumerate(cov_vars):
            if isCorrArr[j] == 1:
                corrVal = corrVal - df[tmpCovVar] * res.params[tmpCovVar]
        df_out[tmp_out_var + '_' + out_suff] = corrVal
        
    out_code = 0
    out_msg = 'Created covariate adjusted data'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}


def DataAdjCov2(df, key_var, target_vars, cov_corr_vars, cov_keep_vars=[], 
               sel_col='', sel_vals = [], out_suff = 'COVADJ'):       
    '''Apply a linear regression model and correct for covariates
    It runs independently for each outcome variable
    The estimation is done on the selected subset and then applied to all samples
    The user can indicate covariates that will be corrected and not
    '''

    if key_var == '':
        out_code = 1        
        out_msg = 'WARNING: Please select primary key variable!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    # Combine covariates (to keep + to correct)
    if cov_keep_vars is []:
        covList = cov_corr_vars;
        isCorr = list(np.ones(len(cov_corr_vars)).astype(int))
    else:
        covList = cov_keep_vars + cov_corr_vars;
        isCorr = list(np.zeros(len(cov_keep_vars)).astype(int)) + list(np.ones(len(cov_corr_vars)).astype(int))

    # Prep data
    TH_MAX_NUM_CAT = 20     ## FIXME: This should be a global var
    df_covs = []
    isCorrArr = []
    for i, tmpVar in enumerate(covList):
        ## Detect if var is categorical
        is_num = pd.to_numeric(df[tmpVar].dropna(), errors='coerce').notnull().all()
        if df[tmpVar].unique().shape[0] < TH_MAX_NUM_CAT:
            is_num = False
        ## Create dummy vars for categorical data
        if is_num == False:
            dfDummy = pd.get_dummies(df[tmpVar], prefix=tmpVar, drop_first=True)
            df_covs.append(dfDummy)
            isCorrArr = isCorrArr + list(np.zeros(dfDummy.shape[1]).astype(int)+isCorr[i])
        else:
            df_covs.append(df[tmpVar])
            isCorrArr.append(isCorr[i])
    df_covs = pd.concat(df_covs, axis=1)
    
    ## Get cov names
    cov_vars = df_covs.columns.tolist()
    str_cov_vars = ' + '.join(cov_vars)
    
    ## Get data with all vars
    if sel_vals == []:
        df_out = pd.concat([df[target_vars], df_covs], axis=1)
        df_train = df_out
    else:
        df_out = pd.concat([df[[sel_col] + target_vars], df_covs], axis=1)
        df_train = df_out[df_out[sel_col].isin(sel_vals)]
        
    ## Fit and apply model for each outcome var
    out_vars = []
    for i, tmp_out_var in enumerate(target_vars):

        ## Fit model
        str_model = tmp_out_var + '  ~ ' + str_cov_vars
        mod = sm.ols(str_model, data=df_train)
        res = mod.fit()

        ## Apply model
        corrVal = df_out[tmp_out_var]
        for j, tmpCovVar in enumerate(cov_vars):
            if isCorrArr[j] == 1:
                corrVal = corrVal - df[tmpCovVar] * res.params[tmpCovVar]
        df_out[tmp_out_var + '_' + out_suff] = corrVal
        out_vars.append(tmp_out_var + out_suff)
        
    logger.warning(df_out)
    logger.warning([key_var] + out_vars)
    input()
    
    df_out = df_out[[key_var] + out_vars]
        
    out_code = 0
    out_msg = 'Created covariate adjusted data'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out, 'out_vars' : out_vars}
    
    return df_out, out_vars

## Normalize data by the given variable
def DataPercICV(df, norm_var, out_suff):
    '''Normalize data
    '''

    if norm_var == '':
        out_code = 3        
        out_msg = 'WARNING: Please select column to normalize by!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    try:
        ## Calculate percent ICV
        df_tmp = df.select_dtypes(include=[np.number])
        df_out = 100 * df_tmp.div(df[norm_var], axis=0)

        ## Add suffix
        df_out = df_out.add_suffix('_' + out_suff)
        df_out = pd.concat([df, df_out], axis=1)        

    except:
        out_code = 2        
        out_msg = 'ERROR: Could not perform ICV correction!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    out_code = 0
    out_msg = 'Created normalized data'
    out_vars = []
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out, 'out_vars' : out_vars}

## Normalize data by the given variable
def DataNormalize(df, key_var, target_vars, norm_var, out_suff):
    '''Normalize data
    '''
    if key_var == '':
        out_code = 1        
        out_msg = 'WARNING: Please select primary key variable!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    if len(target_vars) == 0:
        out_code = 2        
        out_msg = 'WARNING: Please select target variables!'
        return {'out_code' : out_code, 'out_msg' : out_msg}

    if norm_var == '':
        out_code = 3        
        out_msg = 'WARNING: Please select column to normalize by!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    df_out = df[norm_var].mean() * df[target_vars].div(df[norm_var], axis=0)
    #df_out = 100 * df[target_vars].div(df[norm_var], axis=0)   
    
    df_out = df_out.add_suffix('_' + out_suff)
    out_vars = df_out.columns.tolist()
    df_out = pd.concat([df[[key_var]], df_out], axis=1)        

    out_code = 0
    out_msg = 'Created normalized data'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out, 'out_vars' : out_vars}


## Drop duplicates
def DataDrop(df, target_vars):
    '''Drop duplicates from data
    '''
    if len(target_vars) == 0:
        out_code = 1        
        out_msg = 'WARNING: Please select variable(s)!'
        return {'out_code' : out_code, 'out_msg' : out_msg}
    
    df_out = df.drop_duplicates(subset = target_vars)
    
    out_code = 0
    out_msg = 'Created data without duplicates'
    return {'out_code' : out_code, 'out_msg' : out_msg, 'df_out' : df_out}

########################################################
## Display widget functions

def WidgetShowTable(widget_in, df = None, dset_name = None):

    ## Read data and user selection
    if df is None:
        dset_name = widget_in.data_model_arr.dataset_names[widget_in.active_index]
        #dset_fname = widget_in.data_model_arr.datasets[widget_in.active_index].file_name
        df = widget_in.data_model_arr.datasets[widget_in.active_index].data
        
    ## Load data to data view 
    widget_in.dataView = QtWidgets.QTableView()
    
    ## Reduce data size to make the app run faster
    df_tmp = df.head(widget_in.data_model_arr.TABLE_MAXROWS)

    ## Round values for display
    df_tmp = df_tmp.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

    widget_in.PopulateTable(df_tmp)

    ## Set data view to mdi widget
    sub = QMdiSubWindow()
    sub.setWidget(widget_in.dataView)
    #sub.setWindowTitle(dset_name + ': ' + os.path.basename(dset_fname))
    sub.setWindowTitle(dset_name)
    widget_in.mdi.addSubWindow(sub)        
    sub.show()
    widget_in.mdi.tileSubWindows()
    

