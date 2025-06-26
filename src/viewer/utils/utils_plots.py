import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_misc as utilmisc

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr

def panel_view_centiles(method, var_type):
    """
    Panel for viewing data centiles
    """
    flag_settings = st.sidebar.checkbox('Hide plot settings')
    ss_sel = st.session_state.selections

    # Add tabs for parameter settings
    with st.container(border=True):
        if not flag_settings:
            ptab1, ptab2, ptab3 = st.tabs(
                ['Data', 'Centiles', 'Plot Settings']
            )        
            with ptab1:
                if var_type == 'rois':
                    ss_sel['yvar'] = panel_select_roi(method, '_centiles')
                
                elif var_type == 'biomarkers':
                    list_vars = ['WM', 'GM', 'VN']
                    ss_sel['yvar'] = st.selectbox(
                        'Select var',
                        list_vars
                    )
                
            with ptab2:
                ss_sel['centile_type'] = panel_select_centile_type()
                ss_sel['centile_values'] = panel_select_centile_values()
                ss_sel['flag_norm_centiles'] = st.checkbox(
                    'Normalize Centiles'
                )

            with ptab3:
                st.session_state.plot_settings["num_per_row"] = st.slider(
                    "Number of plots per row",
                    st.session_state.plot_settings["min_per_row"],
                    st.session_state.plot_settings["max_per_row"],
                    st.session_state.plot_settings["num_per_row"],
                    disabled=False,
                )

                st.session_state.plot_params["h_coeff"] = st.slider(
                    "Plot height",
                    min_value=st.session_state.plot_settings["h_coeff_min"],
                    max_value=st.session_state.plot_settings["h_coeff_max"],
                    value=st.session_state.plot_settings["h_coeff"],
                    step=st.session_state.plot_settings["h_coeff_step"],
                    disabled=False,
                )

                # Checkbox to show/hide plot legend
                st.session_state.plot_params["hide_legend"] = st.checkbox(
                    "Hide legend",
                    value=st.session_state.plot_params["hide_legend"],
                    disabled=False,
                )

    if ss_sel['yvar'] is None:
        return

    # Set plot type to centile
    st.session_state.plot_params['ptype'] = 'scatter'
    st.session_state.plot_params['xvar'] = 'Age'
    st.session_state.plot_params['traces'] = st.session_state.plot_params['centile_values']
    st.session_state.plot_params['method'] = method
    st.session_state.plot_params['yvar'] = ss_sel['yvar']
    st.session_state.plot_params['centile_type'] = ss_sel['centile_type']
    st.session_state.plot_params['centile_values'] = ss_sel['centile_values']
    st.session_state.plot_params['flag_norm_centiles'] = ss_sel['flag_norm_centiles']
        
    # Add buttons to add/delete plots
    c1, c2, c3 = st.sidebar.columns(3, vertical_alignment="center")
    with c1:
        btn_add = st.button("Add Plot")
    with c2:
        btn_del_sel = st.button("Delete Selected")
    with c3:
        btn_del_all = st.button("Delete All")
        
    # Add/delete plot
    if btn_add:
        # Add plot
        st.session_state.plots = add_plot(
            st.session_state.plots,
            st.session_state.plot_params
        )

    if st.session_state.plots.shape[0] == 0:
        # Add plot
        st.session_state.plots = add_plot(
            st.session_state.plots,
            st.session_state.plot_params
        )

    if btn_del_sel:
        st.session_state.plots = delete_sel_plots(
            st.session_state.plots
        )
    if btn_del_all:
        st.session_state.plots = pd.DataFrame(columns=['params'])

    # st.dataframe(st.session_state.plots)
                    
    # Show plots
    show_plots(
        st.session_state.curr_df,
        st.session_state.plots,
        st.session_state.plot_settings,
    )


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
    df_plots.loc[len(df_plots)] = {'params': new_plot_params.copy()}
    return df_plots

def delete_sel_plots(df_plots):
    """
    Removes plots selected by the user
    (removes the row with the given index from the plots dataframe)
    """
    list_sel = []
    for tmp_ind in df_plots.index.tolist():
        if st.session_state[f'_key_plot_sel_{tmp_ind}']:
            list_sel.append(tmp_ind)
            st.session_state[f'_key_plot_sel_{tmp_ind}'] = False

    df_plots = df_plots.drop(list_sel).reset_index().drop(columns=['index'])
    return df_plots

def delete_all_plots(df_plots):
    """
    Removes plots selected by the user
    """
    df_plots = pd.DataFrame(columns=['params'])
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
    # Read centile data
    f_cent = os.path.join(
        st.session_state.paths['centiles'],
        f'{plot_params['method']}_centiles_{plot_params['centile_type']}.csv'
    )
    df_cent = pd.read_csv(f_cent)

    # Main plot
    m = plot_settings["margin"]
    hi = plot_settings["h_init"]
    hc = plot_params["h_coeff"]
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
    utiltr.add_trace_scatter(df, plot_params, plot_settings, fig)

    ## Add linear fit
    #utiltr.add_trace_linreg(df, plot_params, plot_settings, fig)

    ## Add non-linear fit
    #utiltr.add_trace_lowess(df, plot_params, plot_settings, fig)

    ## Add centile trace
    #utiltr.add_trace_centile(df_cent, plot_params, plot_settings, fig)

    #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

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
                if sel_params['ptype'] == "dist": 
                    new_plot = display_dist_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                elif sel_params['ptype'] == "scatter": 
                    new_plot = display_scatter_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                st.checkbox('Select', key = f'_key_plot_sel_{plot_ind}')

    #if st.session_state.plot_params["show_img"]:
    #show_img()

###################################################################
# Panels
def panel_select_var(sel_var_groups, plot_params, var_type, add_none = False):
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

def panel_select_trend(plot_params):
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

def panel_select_centiles(plot_params):
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


def panel_select_settings(plot_params):
    '''
    Panel to select plot settings
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


def panel_set_plot_params(
    plot_params, var_groups_data, var_groups_hue, pipeline
):
    """
    Panel to set plotting parameters
    """
    flag_settings = st.sidebar.checkbox('Hide plot settings')

    # Add tabs for parameter settings
    with st.container(border=True):
        if not flag_settings:
            ptabs = st.tabs(
                ['Data', 'Groups', 'Fit', 'Centiles', 'Plot Settings']
            )
            with ptabs[0]:
                # Select x var
                panel_select_var(var_groups_data, plot_params, 'xvar')
                    
                # Select y var
                panel_select_var(var_groups_data, plot_params, 'yvar')

            with ptabs[1]:
                # Select h var
                panel_select_var(var_groups_hue, plot_params, 'hvar', add_none = True)

            with ptabs[2]:
                panel_select_trend(plot_params)

            with ptabs[3]:
                panel_select_centiles(plot_params)

            with ptabs[4]:
                panel_select_settings(plot_params)
                
    # Set plot type
    plot_params['ptype'] = 'scatter'
    
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

def panel_show_plots():
    '''
    Panel to add/delete/show plots
    '''
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if f'_key_plot_sel_{tmp_ind}' in st.session_state:
            if st.session_state[f'_key_plot_sel_{tmp_ind}']:
                st.write(f'Updated {tmp_ind}')
                st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params

    ## Sidebar options
    cols = st.sidebar.columns(3, vertical_alignment="center")
    with cols[0]:
        btn_add = st.button("Add Plot")
    with cols[1]:
        btn_del_sel = st.button("Delete Selected")
    with cols[2]:
        btn_del_all = st.button("Delete All")

    if btn_add:
        st.session_state.plots = add_plot(
            st.session_state.plots, st.session_state.plot_params.copy()
        )

    # Add a single plot if there is none
    if st.session_state.plots.shape[0] == 0:
        st.session_state.plots = add_plot(
            st.session_state.plots, st.session_state.plot_params.copy()
        )

    if btn_del_sel:
        st.session_state.plots = delete_sel_plots(
            st.session_state.plots
        )
    if btn_del_all:
        st.session_state.plots = delete_all_plots(
            st.session_state.plots
        )

    # Show plots
    show_plots(
        st.session_state.plot_data['df_data'],
        st.session_state.plots,
        st.session_state.plot_settings
    )






