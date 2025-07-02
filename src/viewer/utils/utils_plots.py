import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_user_select as utiluser
import utils.utils_session as utilses
import utils.utils_misc as utilmisc
import utils.utils_mriview as utilmri

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr

import streamlit_antd_components as sac

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)  # or use a large number like 500


def read_data(fdata):
    '''
    Read data file and add column for hue
    '''
    # Read data file
    df = pd.read_csv(fdata)
    
    # Add column to handle hue var = None
    if 'grouping_var' not in df:
        df["grouping_var"] = "Data"
    
    return df

def add_plot(df_plots, new_plot_params):
    """
    Adds a new plot (new row to the plots dataframe)
    """   
    df_plots.loc[len(df_plots)] = {
        'params': new_plot_params.copy(),
        'flag_sel': True
    }
    return df_plots

def delete_sel_plots(df_plots):
    """
    Removes plots selected by the user
    (removes the row with the given index from the plots dataframe)
    """
    list_sel = []
    for tmp_ind in df_plots.index.tolist():
        if st.session_state[f'_flag_sel_{tmp_ind}']:
            list_sel.append(tmp_ind)
            del st.session_state[f'_flag_sel_{tmp_ind}']

    df_plots = df_plots.drop(list_sel).reset_index().drop(columns=['index'])
    return df_plots

def delete_all_plots():
    """
    Removes all plots
    """
    for tmp_ind in st.session_state.plots.index.tolist():
        del st.session_state[f'_flag_sel_{tmp_ind}']
    df_plots = pd.DataFrame(columns=['flag_sel', 'params'])
    return df_plots

def set_x_bounds(df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str, xvar: str) -> None:
    '''
    Set x and y min/max, if not set
    '''
    xmin = df[xvar].min()
    xmax = df[xvar].max()
    dx = xmax - xmin
    if dx == 0:  # Margin defined based on the value if delta is 0
        xmin = xmin - xmin / 8
        xmax = xmax + xmax / 8
    else:  # Margin defined based on the delta otherwise
        xmin = xmin - dx / 5
        xmax = xmax + dx / 5
    df_plots.loc[plot_id, "xmax"] = xmax
    df_plots.loc[plot_id, "xmin"] = xmin

def set_y_bounds(df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str, yvar: str) -> None:
    '''
    Set x and y min/max, if not set
    '''
    ymin = df[yvar].min()
    ymax = df[yvar].max()
    dy = ymax - ymin
    if dy == 0:  # Margin defined based on the value if delta is 0
        ymin = ymin - ymin / 8
        ymax = ymax + ymax / 8
    else:  # Margin defined based on the delta otherwise
        ymin = ymin - dy / 5
        ymax = ymax + dy / 5
    df_plots.loc[plot_id, "ymax"] = ymax
    df_plots.loc[plot_id, "ymin"] = ymin
    

def display_dist_plot(df, plot_params, plot_ind, plot_settings):
    '''
    Display dist plot
    '''
    # Read color map for data
    colors = plot_settings['cmap']['data']

    # Read plot params
    xvar = plot_params["xvar"]
    yvar = plot_params["yvar"]
    hvar = plot_params["hvar"]
    hvals = plot_params["hvals"]
    traces = plot_params['traces']

    # Add a temp column if group var is not set
    dft = df.copy()
    if hvar is None:
        hvar = "All"
        hvals = None
        dft["All"] = "Data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals is None:
        hvals = vals_hue_all

    data = []
    bin_sizes = []
    colors_sel = []
    for hname in hvals:
        col_ind = vals_hue_all.index(hname)  # Select index of colour for the category
        dfh = dft[dft[hvar] == hname]
        x_tmp = dfh[xvar]
        x_range = x_tmp.max() - x_tmp.min()
        bin_size = x_range / binnum
        bin_sizes.append(bin_size)
        data.append(x_tmp)
        colors_sel.append(colors[col_ind])

    show_hist = "histogram" in traces
    show_curve = "density" in traces
    show_rug = "rug" in traces

    fig = ff.create_distplot(
        data,
        hvals,
        histnorm="",
        bin_size=bin_sizes,
        colors=colors_sel,
        show_hist=show_hist,
        show_rug=show_rug,
        show_curve=show_curve,
    )
    return fig

def display_scatter_plot(df, plot_params, plot_ind, plot_settings):
    '''
    Display scatter plot
    '''
    def callback_plot_clicked() -> None:
        """
        Set the active plot id to plot that was clicked
        """
        st.session_state.plot_active = plot_ind

        # Detect MRID from the click info and save to session_state
        hind = utilmisc.get_index_in_list(df.columns.tolist(), curr_params['hvar'])
        
        sel_info = st.session_state[f"bubble_chart_{plot_ind}"]
        if len(sel_info["selection"]["points"]) > 0:
            sind = sel_info["selection"]["point_indices"][0]
            if hind is None:
                sel_mrid = df.iloc[sind]["MRID"]
            else:
                lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                sel_mrid = df[df[curr_params["hvar"]] == lgroup].iloc[sind][
                    "MRID"
                ]
            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, 'params']['yvar']
            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi
            #st.session_state.sel_roi_img = sel_roi
            # st.rerun()

        print(f'Clicked {sel_mrid}')

    curr_params = st.session_state.plots.loc[plot_ind, 'params']

    # Read centile data
    f_cent = os.path.join(
        st.session_state.paths['centiles'],
        f'{plot_params['method']}_centiles_{plot_params['centile_type']}.csv'
    )
    df_cent = pd.read_csv(f_cent)

    # Main plot
    m = plot_settings["margin"]
    hi = plot_settings["h_init"]
    hc = plot_settings["h_coeff"]
    layout = go.Layout(
        height = hi * hc,
        margin = dict(l=m, r=m, t=m, b=m),
    )
    fig = go.Figure(layout=layout)

    # Add axis labels
    fig.update_layout(        
        xaxis_title = plot_params["xvar"], yaxis_title = plot_params["yvar"]
    )
    
    # Add data scatter
    if df is not None:
        utiltr.add_trace_scatter(df, plot_params, plot_settings, fig)

    # Add linear fit
    if df is not None:
        utiltr.add_trace_linreg(df, plot_params, plot_settings, fig)

    # Add non-linear fit
    if df is not None:
        utiltr.add_trace_lowess(df, plot_params, plot_settings, fig)

    # Add centile trace
    if df_cent is not None:
        utiltr.add_trace_centile(df_cent, plot_params, plot_settings, fig)

    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}", on_select=callback_plot_clicked)
    # st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    return fig

def show_plots(df, df_plots, plot_settings):
    """
    Display all plots
    """
    # Read plot ids
    list_plots = df_plots.index.tolist()
    plots_per_row = plot_settings["num_per_row"]

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
    for i, plot_ind in enumerate(list_plots):        
        column_no = i % plots_per_row
        if column_no == 0:
            blocks = st.columns(plots_per_row)
        sel_params = df_plots.loc[plot_ind, 'params']
        with blocks[column_no]:
            with st.container(border=True):
                if sel_params['plot_type'] == "dist": 
                    new_plot = display_dist_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                elif sel_params['plot_type'] == "scatter": 
                    new_plot = display_scatter_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                st.checkbox(
                    'Select',
                    key = f'_flag_sel_{plot_ind}',
                    value = df_plots.loc[plot_ind, 'flag_sel']
                )
                df_plots.loc[plot_ind, 'flag_sel'] = st.session_state[f'_flag_sel_{plot_ind}']

    if st.sidebar.button('sss'):
        st.session_state.plot_settings["show_img"] = True
    
    if st.session_state.plot_settings["show_img"]:
        in_dir = st.session_state.paths['project']
        mri_params = st.session_state.mri_params
        mrid = st.session_state.sel_mrid
        ulay = os.path.join(
            in_dir, 't1', f'{mrid}_T1.nii.gz'
        )
        olay = os.path.join(
            in_dir, 'dlmuse_seg', f'{mrid}_T1_DLMUSE.nii.gz'
        )
        utilmri.panel_view_seg(ulay, olay, mri_params)

###################################################################
# User selections
def user_select_var2(sel_var_groups, plot_params, var_type, add_none = False):
    '''
    User panel to select a variable 
    Variables are grouped in categories
    '''
    df_groups = st.session_state.dicts['df_var_groups'].copy()
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    st.markdown(f'##### Variable: {var_type}')
    cols = st.columns([1,3])
    with cols[0]:
        
        list_group = df_groups.group.unique().tolist()
        try:
            curr_value = plot_params[f'{var_type}_group']
            curr_index = list_group.index(curr_value)
        except ValueError:
            curr_index = 0
            
        st.selectbox(
            "Variable Group",
            list_group,
            key = f'_{var_type}_group',
            index = curr_index
        )
        plot_params[f'{var_type}_group'] = st.session_state[f'_{var_type}_group']

    with cols[1]:

        sel_group = plot_params[f'{var_type}_group']
        if sel_group is None:
            return
        
        sel_atlas = df_groups[df_groups['group'] == sel_group]['atlas'].values[0]
        list_vars = df_groups[df_groups['group'] == sel_group]['values'].values[0]
        
        # Convert MUSE ROI variables from index to name
        if sel_atlas == 'muse':
            roi_dict = st.session_state.dicts['muse']['ind_to_name']
            list_vars = [roi_dict[k] for k in list_vars]

        if add_none:
            list_vars = ['None'] + list_vars

        try:
            curr_value = plot_params[var_type]
            curr_index = list_vars.index(curr_value)
        except ValueError:
            curr_index = 0
            
        st.selectbox(
            "Variable Name",
            list_vars,
            key = f'_{var_type}',
            index = curr_index
        )
        
        plot_params[var_type] = st.session_state[f'_{var_type}']

def user_select_var(sel_var_groups, plot_params, var_type, add_none = False):
    '''
    User panel to select a variable grouped in categories
    '''
    df_groups = st.session_state.dicts['df_var_groups'].copy()
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    # Create nested var lists
    sac_items = []
    for tmpg in df_groups.group.unique().tolist():
        tmpl = df_groups[df_groups['group'] == tmpg]['values'].values[0]
        tmp_item = sac.CasItem(tmpg, icon='app', children=tmpl)
        sac_items.append(tmp_item)

    sel = sac.cascader(
        items = sac_items,
        label=f'Variable: {var_type}', index=[0,1], multiple=False, search=True, clear=True
    )
        
    st.write(sel)

def user_select_trend(plot_params):
    '''
    Panel to select trend
    '''
    list_trends = st.session_state.plot_settings["trend_types"]
    try:
        curr_value = plot_params['trend']
        curr_index = list_trends.index(curr_value)
    except ValueError:
        curr_index = 0
    
    st.selectbox(
        "Select trend type",
        options = list_trends,
        key='_sel_trend',
        index = curr_index
    )
    plot_params['trend'] = st.session_state['_sel_trend']

    if plot_params['trend'] is None:
        return

    if plot_params['trend'] == 'None':
        return

    if plot_params['trend'] == 'Linear':
        #if '_show_conf' not in st.session_state:
            #st.session_state['_show_conf'] = plot_params['show_conf']
        st.checkbox(
            "Add confidence interval", 
            key='_show_conf',
            value = plot_params['show_conf']
        )
        plot_params['show_conf'] = st.session_state['_show_conf']

    elif plot_params['trend'] == 'Smooth LOWESS Curve':
        #if '_lowess_s' not in st.session_state:
            #st.session_state['_lowess_s'] = plot_params['lowess_s']
        st.slider(
            "Smoothness",
            min_value=0.4,
            max_value=1.0,
            step=0.1,
            key = '_lowess_s',
            value=plot_params['lowess_s'],
        )
        plot_params['lowess_s'] = st.session_state['_lowess_s']

def user_select_centiles(plot_params):
    '''
    User panel to select centile values
    '''
    #FIXME (move to session state)
    list_types = ['None', 'CN', 'CN-Females', 'CN-Males', 'CN-ICVNorm']
    list_values = ['centile_5', 'centile_25', 'centile_50', 'centile_75', 'centile_95']

    ## Select centile type
    try:
        curr_value = plot_params['centile_type']
        curr_index = list_types.index(curr_value)
    except ValueError:
        curr_index = 0
    
    #if '_centile_type' not in st.session_state:
        #st.session_state['_centile_type'] = plot_params['centile_type']
    st.selectbox(
        "Centile Type",
        list_types,
        key = '_centile_type',
        index = curr_index
    )
    plot_params['centile_type'] = st.session_state['_centile_type']
    
    if plot_params['centile_type'] is None:
        return

    if plot_params['centile_type'] == 'None':
        return

    ## Select centile values
    #if '_centile_values' not in st.session_state:
        #st.session_state['_centile_values'] = plot_params['centile_values']
    st.multiselect(
        "Centile Values",
        list_values,
        key = '_centile_values',
        default = plot_params['centile_values']
    )
    plot_params['centile_values'] = st.session_state['_centile_values']

def user_select_plot_settings(plot_params):
    '''
    Panel to select plot args from the user
    '''
    st.session_state.plot_settings["num_per_row"] = st.slider(
        "Number of plots per row",
        st.session_state.plot_settings["min_per_row"],
        st.session_state.plot_settings["max_per_row"],
        st.session_state.plot_settings["num_per_row"],
        disabled=False,
    )

    plot_params["h_coeff"] = st.slider(
        "Plot height",
        min_value=st.session_state.plot_settings["h_coeff_min"],
        max_value=st.session_state.plot_settings["h_coeff_max"],
        value=st.session_state.plot_settings["h_coeff"],
        step=st.session_state.plot_settings["h_coeff_step"],
        disabled=False,
    )

    # Checkbox to show/hide plot legend
    plot_params["hide_legend"] = st.checkbox(
        "Hide legend",
        value=st.session_state.plot_settings["hide_legend"],
        disabled=False,
    )

def panel_set_params_plot(plot_params, var_groups_data, var_groups_hue, pipeline):
    """
    Panel to set plotting parameters
    """
    # Add tabs for parameter settings
    with st.container(border=True):
        flag_settings = True  #FIXME
        if not flag_settings:
            tab = sac.tabs(
                items=[
                    sac.TabsItem(label='Data'),
                    sac.TabsItem(label='Groups'),
                    sac.TabsItem(label='Fit'),
                    sac.TabsItem(label='Centiles'),
                    sac.TabsItem(label='Plot Settings')
                ],
                size='sm',
                align='left'
            )
            
            if tab == 'Data':
                # Select x var
                user_select_var(var_groups_data, plot_params, 'xvar')
                    
                # Select y var
                user_select_var(var_groups_data, plot_params, 'yvar')

            elif tab == 'Groups':
                # Select h var
                user_select_var(var_groups_hue, plot_params, 'hvar', add_none = True)

            elif tab == 'Fit':
                user_select_trend(plot_params)

            elif tab == 'Centiles':
                user_select_centiles(plot_params)

            elif tab == 'Plot Settings':
                user_select_plot_settings(plot_params)
                
    # Set plot type
    plot_params['plot_type'] = 'scatter'
    
    # Set plot traces
    plot_params['traces'] = ['data']

    if plot_params['centile_values'] is not None:
        plot_params['traces'] = plot_params['traces'] + plot_params['centile_values']

    if plot_params['trend'] == 'Linear':
        plot_params['traces'] = plot_params['traces'] + ['lin_fit']

    if plot_params['show_conf']:
        plot_params['traces'] = plot_params['traces'] + ['conf_95%']

    if plot_params['trend'] == 'Smooth LOWESS Curve':
        plot_params['traces'] = plot_params['traces'] + ['lowess']
        
    plot_params['method'] = pipeline
    plot_params['flag_norm_centiles'] = False

def panel_set_params_centile_plot(
    plot_params, var_groups_data, pipeline, flag_hide_settings = False
):
    """
    Panel to select centile plot args from the user
    """    
    # Add tabs for parameter settings
    with st.container(border=True):
        if not flag_hide_settings:
            tab = sac.tabs(
                items=[
                    sac.TabsItem(label='Data'),
                    sac.TabsItem(label='Centiles'),
                    sac.TabsItem(label='Plot Settings')
                ],
                size='sm',
                align='left'
            )
            if tab == 'Data':
                # Select x var
                sel_var = utiluser.select_var_from_group(
                    st.session_state.dicts['df_var_groups'],
                    ['age'],
                    'xvar', 
                    st.session_state.plot_params['xvargroup'],
                    False,
                    st.session_state.dicts['muse']['ind_to_name']
                )
                st.session_state.plot_params['xvargroup'] = sel_var
                st.session_state.plot_params['xvar'] = sel_var[1]
                
                # Select y var
                sel_var = utiluser.select_var_from_group(
                    st.session_state.dicts['df_var_groups'],
                    ['roi'],
                    'yvar', 
                    st.session_state.plot_params['yvargroup'],
                    False,
                    st.session_state.dicts['muse']['ind_to_name']
                )
                st.session_state.plot_params['yvargroup'] = sel_var
                st.session_state.plot_params['yvar'] = sel_var[1]

            elif tab == 'Centiles':
                user_select_centiles(plot_params)

            elif tab == 'Plot Settings':
                user_select_plot_settings(plot_params)
                
    # Set plot type
    plot_params['plot_type'] = 'scatter'
    
    # Set plot traces
    plot_params['traces'] = ['data']

    if plot_params['centile_values'] is not None:
        plot_params['traces'] = plot_params['traces'] + plot_params['centile_values']
        
    plot_params['method'] = pipeline
    plot_params['flag_norm_centiles'] = False


def panel_show_plots():
    '''
    Panel to show plots
    '''
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if st.session_state.plots.loc[tmp_ind, 'flag_sel']:
            st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params.copy()

    ## Sidebar options
    with st.sidebar:
        
        sac.divider(label='Actions', icon = 'arrow-right-circle', align='center', color='gray')
        
        flag_settings = st.sidebar.checkbox('Hide plot settings')
        
        cols = st.columns([2,3,2])
        with cols[0]:
            if st.button('Add Plot'):
                st.session_state.plots = add_plot(
                    st.session_state.plots, st.session_state.plot_params
                )
        
        # Add a single plot if there is none
        if st.session_state.plots.shape[0] == 0:
            st.session_state.plots = add_plot(
                st.session_state.plots, st.session_state.plot_params
            )
        
        with cols[1]:
            if st.button('Delete Selected'):
                st.session_state.plots = delete_sel_plots(
                    st.session_state.plots
                )
        
        with cols[2]:
            if st.button('Delete All'):
                st.session_state.plots = delete_all_plots()

    # Show plots
    show_plots(
        st.session_state.plot_data['df_data'],
        st.session_state.plots,
        st.session_state.plot_settings
    )

def panel_show_centile_plots():
    '''
    Panel to show centile plots
    '''
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if st.session_state.plots.loc[tmp_ind, 'flag_sel']:
            st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params.copy()

    ## Sidebar options
    with st.sidebar:
        
        sac.divider(label='Actions', icon = 'arrow-right-circle', align='center', color='gray')
        
        flag_settings = st.sidebar.checkbox('Hide plot settings')
        
        cols = st.columns([2,3,2])
        with cols[0]:
            if st.button('Add Plot'):
                st.session_state.plots = add_plot(
                    st.session_state.plots, st.session_state.plot_params
                )
        
        # Add a single plot if there is none
        if st.session_state.plots.shape[0] == 0:
            st.session_state.plots = add_plot(
                st.session_state.plots, st.session_state.plot_params
            )
        
        with cols[1]:
            if st.button('Delete Selected'):
                st.session_state.plots = delete_sel_plots(
                    st.session_state.plots
                )
        
        with cols[2]:
            if st.button('Delete All'):
                st.session_state.plots = delete_all_plots()

    # Show plots
    show_plots(
        None,
        st.session_state.plots,
        st.session_state.plot_settings
    )






