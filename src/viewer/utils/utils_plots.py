import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_trace as utiltr

###################################################################
# Misc utils

def add_items_to_list(my_list: list, items_to_add: list) -> list:
    """Adds multiple items to a list, avoiding duplicates.

    Args:
      my_list: The list to add items to.
      items_to_add: A list of items to add.

    Returns:
      The modified list.
    """
    for item in items_to_add:
        if item not in my_list:
            my_list.append(item)
    return my_list


def remove_items_from_list(my_list: list, items_to_remove: list) -> list:
    """Removes multiple items from a list.

    Args:
      my_list: The list to remove items from.
      items_to_remove: A list of items to remove.

    Returns:
      The modified list.
    """
    out_list = []
    for item in my_list:
        if item not in items_to_remove:
            out_list.append(item)
    return out_list

def get_index_in_list(in_list: list, in_item: str) -> Optional[int]:
    """
    Returns the index of the item in list, or None if item not found
    """
    if in_item not in in_list:
        return None
    else:
        return in_list.index(in_item)

###################################################################
# Traces

def add_trace_scatter(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    # Set colormap
    colors = st.session_state.plot_colors["data"]

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "grouping_var"
        dft["grouping_var"] = "Data"
    vals_hue_all = sorted(dft[hvar].unique())

    if hvals == []:
        hvals = vals_hue_all

    if "data" in plot_params['traces']:
        for hname in hvals:
            
            print(hvals)
            
            col_ind = vals_hue_all.index(hname)  # Select index of colour for the category
            dfh = dft[dft[hvar] == hname]
                        
            trace = go.Scatter(
                x=dfh[plot_params['xvar']],
                y=dfh[plot_params['yvar']],
                mode="markers",
                marker={"color": colors[col_ind]},
                name=hname,
                legendgroup=hname,
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(trace)

        #fig.update_layout(xaxis_range=[xmin, xmax])
        #fig.update_layout(yaxis_range=[ymin, ymax])

def add_trace_linreg(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    """
    Add linear fit and confidence interval
    """
    # Set colormap
    colors = st.session_state.plot_colors["data"]

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    # Calculate fit
    dict_fit = utilstat.linreg_model(
        dft, plot_params['xvar'], plot_params['yvar'], hvar
    )

    # Add traces for the fit and confidence intervals
    if "lin_fit" in plot_params['traces']:
        for hname in hvals:
            col_ind = vals_hue_all.index(
                hname
            )  # Select index of colour for the category
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                mode="lines",
                line={"color": colors[col_ind]},
                name=f"lin_{hname}",
                legendgroup=hname,
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(trace)

    if "conf_95%" in traces:
        for hname in hvals:
            col_ind = vals_hue_all.index(
                hname
            )  # Select index of colour for the category
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            conf_int = dict_fit[hname]["conf_int"]
            trace = go.Scatter(
                x=np.concatenate([x_hat, x_hat[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor=f"rgba({colors[col_ind][4:-1]}, 0.2)",  # Add alpha channel
                line=dict(color=f"rgba({colors[col_ind][4:-1]}, 0)"),
                hoverinfo="skip",
                name=f"lin_conf95_{hname}",
                legendgroup=hname,
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(trace)

    # fig.update_layout(xaxis_range=[xmin, xmax])
    # fig.update_layout(yaxis_range=[ymin, ymax])
    return fig

def add_trace_lowess(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    # Set colormap
    colors = st.session_state.plot_colors["data"]

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    dict_fit = utilstat.lowess_model(
        dft, plot_params['xvar'], plot_params['yvar'], hvar, lowess_s
    )

    # Add traces for the fit and confidence intervals
    for hname in hvals:
        col_ind = vals_hue_all.index(hname)  # Select index of colour for the category
        x_hat = dict_fit[hname]["x_hat"]
        y_hat = dict_fit[hname]["y_hat"]
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            # showlegend=False,
            mode="lines",
            line={"color": colors[col_ind]},
            name=f"lowess_{hname}",
            legendgroup=hname,
            showlegend=not plot_params['hide_legend'],
        )
        fig.add_trace(trace)

    # fig.update_layout(xaxis_range=[xmin, xmax])
    # fig.update_layout(yaxis_range=[ymin, ymax])

def add_trace_dot(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    df_tmp = df[df.MRID == sel_mrid]
    trace = go.Scatter(
        x=df_tmp[plot_params['xvar']],
        y=df_tmp[plot_params['yvar']],
        mode="markers",
        name="Selected",
        marker=dict(
            color="rgba(250, 50, 50, 0.5)", size=12, line=dict(color="Red", width=3)
        ),
        showlegend=not plot_params['hide_legend'],
    )
    fig.add_trace(trace)

def add_trace_centile(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    # Set colormap
    colors = st.session_state.plot_colors["centile"]

    # Get centile values for the selected roi
    df_tmp = df[df.VarName == plot_params['yvar']]

    # Create line traces
    for i, cvar in enumerate(df_tmp.columns[2:]):
        if cvar in plot_params['traces']:
            ctrace = go.Scatter(
                x=df_tmp[plot_params['xvar']],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                legendgroup="centiles",
                line=dict(color=colors[i]),
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(ctrace)  # plot in first row

    # Update min/max
    #fig.update_layout(xaxis_range=[xmin, xmax])
    #fig.update_layout(yaxis_range=[ymin, ymax])

    return fig

def add_trace_dots(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    trace = go.Scatter(
        x=df[plot_params['xvar']],
        y=df[plot_params['yvar']],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace

###################################################################
# Plots

def add_plot(df_plots, new_plot_params):
    """
    Adds a new plot 
    (adds a new row to the plots dataframe with new plot params)
    """
    
    new_plot_params['xvar'] = 'Age'
    new_plot_params['yvar'] = 'GM'
    new_plot_params['traces'] = [
        'centile_5', 'centile_50', 'centile_95'
    ]
    
    df_plots.loc[len(df_plots)] = {'params': new_plot_params}
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
    
    print(f'selected: {list_sel}')
    
    df_plots = df_plots.drop(list_sel).reset_index().drop(columns=['index'])
    return df_plots


def set_x_bounds(
    df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str, xvar: str
) -> None:
    # Set x and y min/max if not set
    # Values include some margin added for viewing purposes
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


def set_y_bounds(
    df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str, yvar: str
) -> None:
    # Set x and y min/max if not set
    # Values include some margin added for viewing purposes
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

def dist_plot(df, plot_params, plot_ind):
    # Set colormap
    colors = st.session_state.plot_colors["data"]

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "Data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
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
        # hide_legend=hide_legend  ## THIS IS NOT AVAILABLE IN FF
    )

    return fig

def display_plot(df, plot_params, sel_mrid, plot_ind):
    """
    Display plot
    """
    # Main plot
    m = st.session_state.plot_const["margin"]
    hi = st.session_state.plot_const["h_init"]
    hc = st.session_state.plot_params["h_coeff"]
    layout = go.Layout(
        height = hi * hc,
        margin = dict(l=m, r=m, t=m, b=m),
    )
    fig = go.Figure(layout=layout)

    xvar = plot_params["xvar"]
    yvar = plot_params["yvar"]

    # Add axis labels
    fig.update_layout(xaxis_title = xvar, yaxis_title = yvar)

    if plot_params['ptype'] == 'dist':
        # Main plot
        fig = utiltr.dist_plot(
            df_plots_filt, sel_plot["xvar"],
            sel_plot["hvar"],
            sel_plot["hvals"],
            sel_plot["traces"],
            st.session_state.plot_const["distplot_binnum"],
            st.session_state.plot_params["hide_legend"],
        )
        st.plotly_chart(fig, key=f"key_chart_{plot_id}")

    elif plot_params['ptype'] == 'scatter':

        ## Add data scatter
        #add_trace_scatter(df, plot_params, fig)

        # Add centile trace
        add_trace_centile(df, plot_params, fig)

        #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
        st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    elif plot_params['ptype'] == 'centile':
        # Add centile trace
        add_trace_centile(df, plot_params, fig)

        #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
        st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")


    return fig


def display_scatter_plot(df, plot_params, sel_mrid, plot_ind):
    """
    Display plot
    """
    # Main plot
    m = st.session_state.plot_const["margin"]
    hi = st.session_state.plot_const["h_init"]
    hc = st.session_state.plot_params["h_coeff"]
    layout = go.Layout(
        height = hi * hc,
        margin = dict(l=m, r=m, t=m, b=m),
    )
    fig = go.Figure(layout=layout)

    xvar = plot_params["xvar"]
    yvar = plot_params["yvar"]

    # Add axis labels
    fig.update_layout(xaxis_title = xvar, yaxis_title = yvar)

    ## Add data scatter
    #add_trace_scatter(df, plot_params, fig)

    # Add centile trace
    add_trace_centile(df, plot_params, fig)

    #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    return fig

def display_centile_plot(df, plot_params, plot_ind):
    """
    Display plot
    """
    # Main plot
    m = st.session_state.plot_const["margin"]
    hi = st.session_state.plot_const["h_init"]
    hc = st.session_state.plot_params["h_coeff"]
    layout = go.Layout(
        height = hi * hc,
        margin = dict(l=m, r=m, t=m, b=m),
    )
    fig = go.Figure(layout=layout)

    xvar = plot_params["xvar"]
    yvar = plot_params["yvar"]

    # Add axis labels
    fig.update_layout(xaxis_title = xvar, yaxis_title = yvar)

    ## Add data scatter
    #add_trace_scatter(df, plot_params, fig)

    # Add centile trace
    add_trace_centile(df, plot_params, fig)

    #st.plotly_chart(fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked)
    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    return fig


def display_dist_plot(
    df_plots: pd.DataFrame, plot_id: str, show_settings: bool, sel_mrid: str
) -> Any:
    """
    Displays the plot with the plot_id
    """
    # Main container for the plot
    with st.container(border=True):

        # Tabs for plot parameters
        df_plots_filt = df_plots
        if show_settings:
            df_plots_filt = add_plot_tabs(df_plots, st.session_state.plots, plot_id)

        sel_plot = st.session_state.plots.loc[plot_id]

        # Main plot
        fig = utiltr.dist_plot(
            df_plots_filt,
            sel_plot["xvar"],
            sel_plot["hvar"],
            sel_plot["hvals"],
            sel_plot["traces"],
            st.session_state.plot_const["distplot_binnum"],
            st.session_state.plot_params["hide_legend"],
        )

        fig.update_layout(
            # height=st.session_state.plot_const['h_init']
            height=st.session_state.plot_const["h_init"]
            * st.session_state.plot_params["h_coeff"],
            margin=dict(
                l=st.session_state.plot_const["margin"],
                r=st.session_state.plot_const["margin"],
                t=st.session_state.plot_const["margin"],
                b=st.session_state.plot_const["margin"],
            ),
        )
        st.plotly_chart(fig, key=f"key_chart_{plot_id}")

        return fig

def show_img3D(
    img: np.ndarray,
    scroll_axis: Any,
    sel_axis_bounds: Any,
    img_name: str,
    size_auto: bool,
) -> None:
    """
    Display a 3D img
    """

    # Create a slider to select the slice index
    slice_index = st.slider(
        f"{img_name}",
        0,
        sel_axis_bounds[1] - 1,
        value=sel_axis_bounds[2],
        key=f"slider_{img_name}",
    )

    # Extract the slice and display it
    if size_auto:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], use_container_width=True)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], use_container_width=True)
        else:
            st.image(img[:, :, slice_index], use_container_width=True)
    else:
        w_img = (
            st.session_state.mriview_const["w_init"]
            * st.session_state.mriview_var["w_coeff"]
        )
        if scroll_axis == 0:
            # st.image(img[slice_index, :, :], use_container_width=True)
            st.image(img[slice_index, :, :], width=w_img)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width=w_img)
        else:
            st.image(img[:, :, slice_index], width=w_img)

def show_plots(df, df_plots):
    """
    Display all plots
    """
    # Read plot ids
    list_plots = df_plots.index.tolist()
    plots_per_row = st.session_state.plot_const["num_per_row"]

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot

    if df.shape[0] == 0:
        st.warning("Dataframe is empty, skip plotting!")
        return

    plots_arr = []
    for i, plot_ind in enumerate(list_plots):
        column_no = i % plots_per_row
        if column_no == 0:
            blocks = st.columns(plots_per_row)
        sel_params = df_plots.loc[plot_ind, 'params']
        with blocks[column_no]:
            with st.container(border=True):
                plot_type = sel_params['plot_type']
                if plot_type == "Scatter Plot":
                    new_plot = display_scatter_plot(
                        df, sel_params, None, plot_ind
                    )
                st.checkbox('Select', key = f'_key_plot_sel_{plot_ind}')
                plots_arr.append(new_plot)
    
    #if st.session_state.plot_params["show_img"]:
    #show_img()


def panel_data_plots(df):
    """
    Panel for adding multiple plots with configuration options
    """
    flag_settings = st.sidebar.checkbox('Hide plot settings')
    flag_params = st.sidebar.checkbox('Hide plot params')

    # Add parameters
    with st.container(border=True):
        
        if not flag_settings:
            ptab1, ptab2, ptab3 = st.tabs(
                ['Plot Settings', 'Variables', 'Others']
            )        
            with ptab1:
                plot_type = st.selectbox(
                    "Plot Type", ["Scatter Plot", "Distribution Plot"], index=0
                )
                if plot_type is not None:
                    st.session_state.plot_params["plot_type"] = plot_type

                st.session_state.plot_const["num_per_row"] = st.slider(
                    "Plots per row",
                    st.session_state.plot_const["min_per_row"],
                    st.session_state.plot_const["max_per_row"],
                    st.session_state.plot_const["num_per_row"],
                    disabled=False,
                )

                st.session_state.plot_params["h_coeff"] = st.slider(
                    "Plot height",
                    min_value=st.session_state.plot_const["h_coeff_min"],
                    max_value=st.session_state.plot_const["h_coeff_max"],
                    value=st.session_state.plot_const["h_coeff"],
                    step=st.session_state.plot_const["h_coeff_step"],
                    disabled=False,
                )

                # Checkbox to show/hide plot legend
                st.session_state.plot_params["hide_legend"] = st.checkbox(
                    "Hide legend",
                    value=st.session_state.plot_params["hide_legend"],
                    disabled=False,
                )

            with ptab2:

                # Selected id
                list_mrid = df.MRID.sort_values().tolist()
                
                st.session_state.sel_mrid = df.MRID.values[0]
                if st.session_state.sel_mrid == "":
                    sel_ind = None
                else:
                    sel_ind = list_mrid.index(st.session_state.sel_mrid)
                sel_mrid = st.selectbox(
                    "Selected subject",
                    list_mrid,
                    sel_ind,
                    help="Select a subject from the list, or by clicking on data points on the plots",
                )
                if sel_mrid is not None:
                    st.session_state.sel_mrid = sel_mrid
                    st.session_state.paths["sel_img"] = ""
                    st.session_state.paths["sel_seg"] = ""

    c1, c2, c3 = st.columns(3, border=True)
    with c1:
        btn_add = st.sidebar.button("Add Plot")
    with c2:
        btn_del_sel = st.sidebar.button("Delete Selected")
    with c3:
        btn_del_all = st.sidebar.button("Delete All")
        
    if btn_add:
        # Add plot
        st.session_state.plots = add_plot(
            st.session_state.plots,
            st.session_state.plot_params
        )

    if btn_del_sel:
        # Add plot
        st.session_state.plots = delete_sel_plots(
            st.session_state.plots
        )

    if btn_del_alll:
        # Add plot
        st.session_state.plots = pd.DataFrame(columns=['params'])
                    
    # Show plot
    show_plots(
        st.session_state.curr_df,
        st.session_state.plots            
    )

def panel_centile_plots(df):
    """
    Panel for adding multiple centile plots with configuration options
    """
    flag_settings = st.sidebar.checkbox('Hide plot settings')

    # Add parameters
    with st.container(border=True):
        
        if not flag_settings:
            ptab1, ptab2, = st.tabs(
                ['Settings', 'Variables']
            )        
            with ptab1:
                st.session_state.plot_const["num_per_row"] = st.slider(
                    "Plots per row",
                    st.session_state.plot_const["min_per_row"],
                    st.session_state.plot_const["max_per_row"],
                    st.session_state.plot_const["num_per_row"],
                    disabled=False,
                )

                st.session_state.plot_params["h_coeff"] = st.slider(
                    "Plot height",
                    min_value=st.session_state.plot_const["h_coeff_min"],
                    max_value=st.session_state.plot_const["h_coeff_max"],
                    value=st.session_state.plot_const["h_coeff"],
                    step=st.session_state.plot_const["h_coeff_step"],
                    disabled=False,
                )

                # Checkbox to show/hide plot legend
                st.session_state.plot_params["hide_legend"] = st.checkbox(
                    "Hide legend",
                    value=st.session_state.plot_params["hide_legend"],
                    disabled=False,
                )

            with ptab2:
                # Select roi
                list_roi = ['GM', 'WM', 'VN']
                sel_roi = st.selectbox(
                    "Select ROI",
                    list_roi,
                    None,
                    help="Select an ROI from the list"
                )
                if sel_roi is not None:
                    st.session_state.plot_params['yvar'] = sel_roi


                # if st.session_state.sel_roi_img == "":
                #     sel_ind = None
                # else:
                #     sel_ind = list_roi.index(st.session_state.sel_roi_img)
                # sel_roi_img = st.selectbox(
                #     "Selected ROI", list_roi, sel_ind, help="Select an ROI from the list"
                # )
                # if sel_roi_img is not None:
                #     st.session_state.sel_roi_img = sel_roi_img



    st.session_state.plot_params["ptype"] = 'centile'

    c1, c2, c3 = st.sidebar.columns(3, vertical_alignment="center")
    with c1:
        btn_add = st.button("Add Plot")
    with c2:
        btn_del_sel = st.button("Delete Selected")
    with c3:
        btn_del_all = st.button("Delete All")
        
    if btn_add:
        # Add plot
        st.session_state.plots = add_plot(
            st.session_state.plots,
            st.session_state.plot_params
        )

    if btn_del_sel:
        # Add plot
        st.session_state.plots = delete_sel_plots(
            st.session_state.plots
        )

    if btn_del_all:
        # Add plot
        st.session_state.plots = pd.DataFrame(columns=['params'])
                    
    # Show plot
    show_plots(
        st.session_state.curr_df,
        st.session_state.plots            
    )
