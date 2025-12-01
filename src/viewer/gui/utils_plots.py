import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_user_select as utiluser
import utils.utils_session as utilses
import utils.utils_misc as utilmisc
import gui.utils_mriview as utilmri

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr
import utils.utils_css as utilcss
import gui.utils_widgets as utilwd

import streamlit_antd_components as sac

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)  # or use a large number like 500

utilcss.load_css()

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

def display_scatter_plot(df, df_cent, plot_params, plot_ind, plot_settings):
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
        
        # print('-------------------------------------')
        # print(sel_info)
        
        if len(sel_info["selection"]["points"]) > 0:
            sind = sel_info["selection"]["point_indices"][0]
            if hind is None:
                sel_mrid = df.iloc[sind]["MRID"]
            else:
                if 'legendgroup' in sel_info["selection"]["points"][0]:
                    lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                    sel_mrid = df[df[curr_params["hvar"]] == lgroup].iloc[sind][
                        "MRID"
                    ]
                else:
                    sel_mrid = df.iloc[sind]["MRID"]
                    
            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, 'params']['yvar']
            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi

        # print(f'Clicked {sel_mrid}')

    curr_params = st.session_state.plots.loc[plot_ind, 'params']
        
    # Filter centiles
    if df_cent is None:
        df_cent_filt = None
    else:
        df_cent_filt = df_cent.copy()
        if 'Age' in df_cent:
            df_cent_filt = df_cent_filt[(df_cent_filt.Age >= plot_params['filter_age'][0]) & (df_cent_filt.Age <= plot_params['filter_age'][1])]

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
    if plot_params['trend'] == 'Linear':
        if df is not None:
            utiltr.add_trace_linreg(df, plot_params, plot_settings, fig)

    # Add non-linear fit
    if plot_params['trend'] == 'Smooth LOWESS Curve':
        if df is not None:
            utiltr.add_trace_lowess(df, plot_params, plot_settings, fig)

    # Add centile trace
    if df_cent_filt is not None:
        utiltr.add_trace_centile(df_cent_filt, plot_params, plot_settings, fig)

    # Add selected dot
    if df is not None:
        sel_mrid = st.session_state.sel_mrid
        if sel_mrid is not None:
            utiltr.add_trace_dot(df, sel_mrid, plot_params, plot_settings, fig)

    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}", on_select=callback_plot_clicked)
    # st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    ## FIXME
    #with st.expander('temp debug'):
        #st.write(df)
        #st.write(df_cent)
        #st.write(curr_params)

    return fig

def show_plots(df, df_cent, df_plots, plot_settings):
    """
    Display data plots
    """
    # Read plot ids
    list_plots = df_plots.index.tolist()
        
    # Set number of plots in each row
    num_plot = len(list_plots)
    if plot_settings['flag_auto']:
        plots_per_row = int(np.min([num_plot, plot_settings["num_per_row"]]))
    else:
        plots_per_row = plot_settings["num_per_row"]

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
    for i, plot_ind in enumerate(list_plots):        
        column_no = i % plots_per_row
        if column_no == 0:
            blocks = st.columns(plots_per_row)
        sel_params = df_plots.loc[plot_ind, 'params']
                
        # Filter data
        if df is None:
            df_filt = None
        else:
            df_filt = df.copy()
            if 'Sex' in df:
                df_filt = df_filt[df_filt.Sex.isin(sel_params['filter_sex'])]
            if 'Age' in df:
                df_filt = df_filt[(df_filt.Age >= sel_params['filter_age'][0]) & (df_filt.Age <= sel_params['filter_age'][1])]

        with blocks[column_no]:
            if sel_params['plot_type'] == "dist": 
                new_plot = display_dist_plot(df_filt, sel_params, plot_ind, plot_settings)
                
            elif sel_params['plot_type'] == "scatter": 
                new_plot = display_scatter_plot(df_filt, df_cent, sel_params, plot_ind, plot_settings)
                
            st.checkbox(
                'Select',
                key = f'_flag_sel_{plot_ind}',
                value = df_plots.loc[plot_ind, 'flag_sel']
            )
            df_plots.loc[plot_ind, 'flag_sel'] = st.session_state[f'_flag_sel_{plot_ind}']

#def show_mriplot():
    #'''
    #Display mri plot
    #'''
    #mrid = st.session_state.sel_mrid
    #if mrid is None:
        #return

    #if st.session_state.plot_settings["flag_hide_mri"] == 'Hide':
        #return

    #in_dir = st.session_state.paths['project']
    #plot_params = st.session_state.plot_params
    #ulay = os.path.join(
        #in_dir, 't1', f'{mrid}_T1.nii.gz'
    #)
    #olay = os.path.join(
        #in_dir, 'DLMUSE_seg', f'{mrid}_T1_DLMUSE.nii.gz'
    #)
    
    #if not os.path.exists(ulay):
        #return

    #if not os.path.exists(olay):
        #return
    
    #utilmri.panel_view_seg(ulay, olay, plot_params)

def add_new_plot(plot_params):
    '''
    Panel to select plot args from the user
    '''
    with st.container(horizontal=True, horizontal_alignment="center"):
        b1 = st.button('Add Plot')
        b2 = st.button('Delete Selected')
        b3 = st.button('Delete All')

        if b1:
            st.session_state.plots = add_plot(
                st.session_state.plots, st.session_state.plot_params
            )

        if b2:
            st.session_state.plots = delete_sel_plots(
                st.session_state.plots
            )

        if b3:
            st.session_state.plots = delete_all_plots()

def panel_show_plots():
    '''
    Panel to show plots
    '''
    pipeline = st.session_state.general_params['sel_pipeline']
    
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if st.session_state.plots.loc[tmp_ind, 'flag_sel']:
            st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params.copy()

    # Show plots
    show_plots(
        st.session_state.plot_data['df_data'],
        st.session_state.plot_data['df_cent'],
        st.session_state.plots,
        st.session_state.plot_settings
    )
    
    if pipeline == 'dlmuse':

        if st.session_state.sel_mrid is not None:
            sac.divider(label='Image Viewer', align='center', color='grape', size = 'lg')

            if st.checkbox('Show MRI?'):
                sel_mrid = st.session_state.sel_mrid
                #######################
                ## Set olay ulay images
                fname = os.path.join(
                    st.session_state.paths['curr_data'], 't1', f'{sel_mrid}_T1.nii.gz'
                )
                if not os.path.exists(fname):
                    st.session_state.mriplot_params['ulay'] = None
                    st.warning(f'Could not find underlay image: {fname}')
                else:
                    st.session_state.mriplot_params['ulay'] = fname

                fname = os.path.join(
                    st.session_state.paths['curr_data'], 'DLMUSE_seg', f'{sel_mrid}_T1_DLMUSE.nii.gz'
                )
                if not os.path.exists(fname):
                    st.session_state.mriplot_params['olay'] = None
                    st.warning(f'Could not find overlay image: {fname}')
                else:
                    st.session_state.mriplot_params['olay'] = fname
                st.session_state.mriplot_params['sel_mrid'] = sel_mrid

                st.session_state.mriplot_params['sel_roi'] = st.session_state.plot_params['yvar']
                
                utilmri.panel_view_seg()






