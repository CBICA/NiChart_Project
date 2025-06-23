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

# def delete_sel_plot(df_plots, list_sel):
#     """
#     Removes plots selected by the user
#     """
#     df_plots = df_plots.drop(list_sel).reset_index().drop(columns=['index'])
#     return df_plots

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

    xvar = plot_params["xvar"]
    yvar = plot_params["yvar"]

    # Add axis labels
    fig.update_layout(xaxis_title = xvar, yaxis_title = yvar)

    # Add data scatter
    if df is not None:
        utiltr.add_trace_scatter(df, plot_params, plot_settings, fig)

    # Add centile trace
    utiltr.add_trace_centile(df_cent, plot_params, plot_settings, fig)

    #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    return fig

def show_plots(df, df_plots, plot_settings):
    """
    Display all plots
    """
    # Read plot ids
    list_plots = df_plots.index.tolist()
    plots_per_row = st.session_state.plot_settings["num_per_row"]

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
    for i, plot_ind in enumerate(list_plots):        
        column_no = i % plots_per_row
        if column_no == 0:
            blocks = st.columns(plots_per_row)
        sel_params = df_plots.loc[plot_ind, 'params']
        with blocks[column_no]:
            border = False
            if st.session_state.plot_settings['sel_plot'] == plot_ind:
                border = True
            with st.container(border=border):
                if sel_params['ptype'] == "dist": 
                    new_plot = display_dist_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                elif sel_params['ptype'] == "scatter": 
                    new_plot = display_scatter_plot(
                        df, sel_params, plot_ind, plot_settings
                    )
                st.checkbox('Select', key = f'_key_plot_sel_{plot_ind}')
                # if st.button('Select', key = f'_key_plot_sel_{plot_ind}'):
                #     st.success('Selected!')
                #     st.session_state.plot_settings['sel_plot'] = plot_ind

    #if st.session_state.plot_params["show_img"]:
    #show_img()

###################################################################
# Panels
def panel_select_roi_vtmp(method, key):
    '''
    User panel to select an ROI
    '''
    ## MUSE ROIs
    if method == 'muse' or method == 'dlmuse':

        # Read dictionaries
        df_derived = st.session_state.rois['muse']['df_derived']

        # df_groups = st.session_state.rois['muse']['df_groups']
        df_groups = st.session_state.dicts['df_var_groups']

        col1, col2 = st.columns([1,3])

        # Select roi group
        with col1:
            list_group = df_groups.group.unique()
            sel_ind = utilmisc.get_index_in_list(
                list_group, st.session_state.selections['sel_roi_group']
            )
            sel_group = st.selectbox(
                "ROI Group",
                list_group,
                sel_ind,
                help="Select ROI group",
                key = f'_sel_roigroup_{key}'
            )
            if sel_group is None:
                return None

        # Select roi
        with col2:
            sel_indices = df_groups[df_groups['group'] == sel_group]['vars'].values[0]

            list_roi = df_derived[df_derived.Index.isin(sel_indices)].Name.tolist()
            sel_ind = utilmisc.get_index_in_list(
                list_roi, st.session_state.selections['sel_roi']
            )
            sel_roi = st.selectbox(
                "ROI Name",
                list_roi,
                sel_ind,
                help="Select an ROI from the list",
                key = f'_sel_roiname_{key}'
            )
            if sel_group is None:
                return None

            st.session_state.selections['sel_roi_group'] = sel_group
            st.session_state.selections['sel_roi'] = sel_roi

        return sel_roi

    elif method == 'dlwmls':
        sel_roi = 'WML'
        return sel_roi

def panel_select_var(sel_var_groups, key):
    '''
    User panel to select a variable
    '''
    # Read dictionaries
    df_derived = st.session_state.rois['muse']['df_derived']
    df_groups = st.session_state.dicts['df_var_groups'].copy()

    # Select var groups
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    col1, col2 = st.columns([1,3])

    # Select roi group
    with col1:
        list_group = df_groups.group.unique()
        sel_ind = utilmisc.get_index_in_list(
            list_group, st.session_state.selections['sel_roi_group']
        )
        sel_group = st.selectbox(
            "ROI Group",
            list_group,
            sel_ind,
            help="Select ROI group",
            key = f'_sel_roigroup_{key}'
        )
        if sel_group is None:
            return None

    # Select roi
    with col2:
        sel_indices = df_groups[df_groups['group'] == sel_group]['values'].values[0]
        list_roi = df_derived[df_derived.Index.isin(sel_indices)].Name.tolist()

        # st.write(sel_indices)
        # st.write(list_roi)

        sel_ind = utilmisc.get_index_in_list(
            list_roi, st.session_state.selections['sel_roi']
        )
        sel_roi = st.selectbox(
            "ROI Name",
            list_roi,
            sel_ind,
            help="Select an ROI from the list",
            key = f'_sel_roiname_{key}'
        )
        if sel_roi is None:
            return None

        st.session_state.selections['sel_roi_group'] = sel_group
        st.session_state.selections['sel_roi'] = sel_roi

    return sel_roi

def panel_select_var_v0(list_vars, sel_var, key):
    '''
    User panel to select a variable
    '''
    sel_ind = utilmisc.get_index_in_list(list_vars, sel_var)
    sel_var = st.selectbox(
        "Variable",
        list_vars,
        sel_ind,
        help="Select a variable from the list",
        key = f'_sel_var_{key}'
    )
    return sel_var

def panel_select_centile_type():
    '''
    User panel to select centile type
    '''
    list_types = ['CN', 'CN-Female', 'CN-Males', 'CN-ICVNorm']
    sel_ind = list_types.index(st.session_state.plot_params['centile_type'])
    sel_type = st.selectbox(
        "Centile Type",
        list_types,
        sel_ind,
        help="Select Centile Type"
    )
    if sel_type is None:
        return None
    st.session_state.plot_params['centile_type'] = sel_type
    
    return sel_type

def panel_select_centile_values():
    '''
    User panel to select centile values
    '''
    list_values = [
        'centile_5', 'centile_25', 'centile_50', 'centile_75', 'centile_95',
    ]
    sel_vals = st.multiselect(
        "Centile Values",
        list_values,
        st.session_state.plot_params['centile_values'],
        help="Select Centile Values"
    )

    if sel_vals is None:
        return None
    st.session_state.plot_params['centile_values'] = sel_vals
    
    return sel_vals

def panel_view_data_v0():
    """
    Panel to plot data variables in the input dataframe
    """
    df_data = st.session_state.plot_data['df_data']
    df_cent = st.session_state.plot_data['df_cent']
    
    if df_data is not None:
        list_vars = df_data.columns.tolist()
    elif df_cent is not None:
        list_vars = df_cent.columns.tolist()
    else:
        list_vars = []
    
    flag_settings = st.sidebar.checkbox('Hide plot settings')
    ss_sel = st.session_state.selections

    # Add tabs for parameter settings
    with st.container(border=True):
        if not flag_settings:
            ptab1, ptab2, ptab3 = st.tabs(
                ['Data', 'Centiles', 'Plot Settings']
            )        
            with ptab1:
                ss_sel['yvar'] = panel_select_var(list_vars, None, '_data')
                
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
    st.session_state.plot_params['traces'] = ['data'] + st.session_state.plot_params['centile_values']
    st.session_state.plot_params['method'] = None
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

    print('plots info')
    print(st.session_state.plots.params[0])
    print('plots info2')

    if btn_del_sel:
        st.session_state.plots = delete_sel_plot(
            st.session_state.plots
        )
    if btn_del_all:
        st.session_state.plots = delete_sel_plot(
            st.session_state.plots
        )

        st.session_state.plots = pd.DataFrame(columns=['params'])

    # st.dataframe(st.session_state.plots)
                    
    # Show plots
    show_plots(
        df_data, st.session_state.plots, st.session_state.plot_settings
    )

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

    print('plots info')
    print(st.session_state.plots.params[0])
    print('plots info2')


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



def panel_set_plot_params(var_groups, pipeline):
    """
    Panel to set plotting parameters
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
                ss_sel['yvar'] = panel_select_var(var_groups, '_data')

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
    st.session_state.plot_params['traces'] = ['data'] + st.session_state.plot_params['centile_values']
    st.session_state.plot_params['method'] = pipeline
    st.session_state.plot_params['yvar'] = ss_sel['yvar']
    st.session_state.plot_params['centile_type'] = ss_sel['centile_type']
    st.session_state.plot_params['centile_values'] = ss_sel['centile_values']
    st.session_state.plot_params['flag_norm_centiles'] = ss_sel['flag_norm_centiles']

def panel_show_plots():
    '''
    Panel to add/delete/show plots
    '''
    # Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if f'_key_plot_sel_{tmp_ind}' in st.session_state:
            if st.session_state[f'_key_plot_sel_{tmp_ind}']:
                st.write(f'updated {tmp_ind}')
                st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params

    c1, c2, c3 = st.sidebar.columns(3, vertical_alignment="center")
    with c1:
        btn_add = st.button("Add Plot")
    with c2:
        btn_del_sel = st.button("Delete Selected")
    with c3:
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

