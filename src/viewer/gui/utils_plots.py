import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_user_select as utiluser
import utils.utils_session as utilses
import utils.utils_misc as utilmisc
import utils.utils_mriview as utilmri

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr
import utils.utils_css as utilcss

import streamlit_antd_components as sac

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)  # or use a large number like 500

utilcss.load_css()

def select_sex_var():
    '''
    Set sex var values
    '''
    sel_vals = st.pills(
        label='Select Sex',
        options=['F', 'M'],
        key = '_sex_var',
        default = st.session_state.plot_params['filter_sex'],
        selection_mode = 'multi',
    )
    st.session_state.plot_params['filter_sex'] = sel_vals

def select_age_range():
    '''
    Set age range values
    '''
    if '_age_range' not in st.session_state:
        st.session_state['_age_range'] = st.session_state.plot_params['filter_age']

    def update_age_range():
        st.session_state.plot_params['filter_age'] = st.session_state['_age_range']

    st.slider(
        'Select Age Range:',
        min_value = st.session_state.plot_settings['min_age'],
        max_value = st.session_state.plot_settings['max_age'],
        key = '_age_range',
        on_change = update_age_range
    )
        
def sidebar_flag_hide_setting():
    '''
    Set flag for hiding the settings
    '''
    if '_flag_hide_settings' not in st.session_state:
        st.session_state['_flag_hide_settings'] = st.session_state.plot_settings['flag_hide_settings']

    def update_val():
        st.session_state.plot_settings['flag_hide_settings'] = st.session_state['_flag_hide_settings']

    with st.sidebar:
        st.radio(
            'Plot Settings',
            ['Show', 'Hide'],
            key = '_flag_hide_settings',
            horizontal = True,
            on_change = update_val
        )

def sidebar_flag_hide_legend():
    '''
    Set flag for hiding the settings
    '''
    if '_flag_hide_legend' not in st.session_state:
        st.session_state['_flag_hide_legend'] = st.session_state.plot_settings['flag_hide_legend']

    def update_val():
        st.session_state.plot_settings['flag_hide_legend'] = st.session_state['_flag_hide_legend']

    with st.sidebar:
        st.radio(
            'Legend',
            ['Show', 'Hide'],
            key = '_flag_hide_legend',
            horizontal = True,
            on_change = update_val
        )

def sidebar_flag_hide_mri():
    '''
    Set flag for hiding the settings
    '''
    if '_flag_hide_mri' not in st.session_state:
        st.session_state['_flag_hide_mri'] = st.session_state.plot_settings['flag_hide_mri']

    def update_val():
        st.session_state.plot_settings['flag_hide_mri'] = st.session_state['_flag_hide_mri']

    with st.sidebar:
        st.radio(
            'MRI Viewer',
            ['Show', 'Hide'],
            key = '_flag_hide_mri',
            horizontal = True,
            on_change = update_val
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
        
        print('-------------------------------------')
        print(sel_info)
        
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

        print(f'Clicked {sel_mrid}')

    curr_params = st.session_state.plots.loc[plot_ind, 'params']

    # Read centile data
    if plot_params['centile_type'] == 'None':
        df_cent = None
    else:
        try:
            f_cent = os.path.join(
                st.session_state.paths['centiles'],
                f'{plot_params['method']}_centiles_{plot_params['centile_type']}.csv'
            )
            df_cent = pd.read_csv(f_cent)
        except:
            st.warning('Could not read centile data!')
            df_cent = None

    # Filter centiles
    dfcf = df_cent.copy()
    if 'Age' in dfcf:
        dfcf = dfcf[(dfcf.Age >= plot_params['filter_age'][0]) & (dfcf.Age <= plot_params['filter_age'][1])]

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
    if dfcf is not None:
        utiltr.add_trace_centile(dfcf, plot_params, plot_settings, fig)

    # Add selected dot
    if df is not None:
        sel_mrid = st.session_state.sel_mrid
        if sel_mrid is not None:
            utiltr.add_trace_dot(df, sel_mrid, plot_params, plot_settings, fig)

    st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}", on_select=callback_plot_clicked)
    # st.plotly_chart(fig, key=f"bubble_chart_{plot_ind}")

    return fig

def show_plots(df, df_plots, plot_settings):
    """
    Display data plots
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
                
        # Filter data
        if df is None:
            dff = None
        else:
            dff = df.copy()
            if 'Sex' in df:
                dff = dff[dff.Sex.isin(sel_params['filter_sex'])]
            if 'Age' in df:
                dff = dff[(dff.Age >= sel_params['filter_age'][0]) & (dff.Age <= sel_params['filter_age'][1])]
                    
        with blocks[column_no]:
            with st.container(border=True):
                if sel_params['plot_type'] == "dist": 
                    new_plot = display_dist_plot(
                        dff, sel_params, plot_ind, plot_settings
                    )
                elif sel_params['plot_type'] == "scatter": 
                    new_plot = display_scatter_plot(
                        dff, sel_params, plot_ind, plot_settings
                    )
                st.checkbox(
                    'Select',
                    key = f'_flag_sel_{plot_ind}',
                    value = df_plots.loc[plot_ind, 'flag_sel']
                )
                df_plots.loc[plot_ind, 'flag_sel'] = st.session_state[f'_flag_sel_{plot_ind}']


def show_mri():
    '''
    Display mri plot
    '''
    mrid = st.session_state.sel_mrid
    if mrid is None:
        return

    if st.session_state.plot_settings["flag_hide_mri"] == 'Hide':
        return

    in_dir = st.session_state.paths['project']
    plot_params = st.session_state.plot_params
    ulay = os.path.join(
        in_dir, 't1', f'{mrid}_T1.nii.gz'
    )
    olay = os.path.join(
        in_dir, 'DLMUSE_seg', f'{mrid}_T1_DLMUSE.nii.gz'
    )
    utilmri.panel_view_seg(ulay, olay, plot_params)

###################################################################
# User selections
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
    plot_params['flag_hide_legend'] = st.checkbox(
        "Hide legend",
        value=st.session_state.plot_settings['flag_hide_legend'],
        disabled=False,
    )

def user_add_plots(plot_params):
    '''
    Panel to select plot args from the user
    '''

    #def toggle_add_plot():
        #print('toggled')

        #if st.session_state['_key_add_plot'] == 'Add Plot':
            #st.session_state.plots = add_plot(
                #st.session_state.plots, st.session_state.plot_params
            #)

        #if st.session_state['_key_add_plot'] == 'Delete Selected':
            #st.session_state.plots = delete_sel_plots(
                #st.session_state.plots
            #)

        #if st.session_state['_key_add_plot'] == 'Delete All':
            #st.session_state.plots = delete_all_plots()

        #st.session_state['_key_add_plot'] = None

    #options = ['Add Plot', 'Delete Selected', 'Delete All']
    #sel = st.segmented_control(
        #"Plot Control",
        #options,
        #selection_mode="single",
        #on_change = toggle_add_plot,
        #key = '_key_add_plot'
    #)

    with st.container(horizontal=True, horizontal_alignment="center"):
        b1 = st.button('Add Plot')
        b2 = st.button('Delete Selected')
        b3 = st.button('Delete All')


def panel_set_params_plot(plot_params, pipeline, list_vars):
    """
    Panel to set plotting parameters
    """
    if st.session_state.plot_settings['flag_hide_settings'] == 'Hide':
        return

    plot_params['method'] = pipeline
    plot_params['flag_norm_centiles'] = False    

    # Add tabs for parameter settings
    #with st.container(border=True):
    with st.expander('Plot Settings'):
        tab = sac.tabs(
            items=[
                sac.TabsItem(label='Data'),
                sac.TabsItem(label='Filters'),
                sac.TabsItem(label='Groups'),
                sac.TabsItem(label='Fit'),
                sac.TabsItem(label='Centiles'),
                sac.TabsItem(label='Plot Settings'),
                #sac.TabsItem(label='Add/Delete Plots')
            ],
            size='sm',
            align='left'
        )

        df_vars = st.session_state.dicts['df_var_groups']
        if tab == 'Data':
            
            # Select x var
            sel_var = utiluser.select_var_from_group(
                'Select x variable:',
                df_vars[df_vars.group.isin(['demog', 'user_data'])],
                plot_params['xvargroup'],
                plot_params['xvar'],
                list_vars,
                flag_add_none = False,
                dicts_rename = {
                    'muse': st.session_state.dicts['muse']['ind_to_name']
                }
            )
            
            if sel_var != []:
                plot_params['xvargroup'] = sel_var[0]
                plot_params['xvar'] = sel_var[1]

            # Select y var
            sel_var = utiluser.select_var_from_group(
                'Select y variable:',
                df_vars[df_vars.category.isin(['demog','roi','biomarker','user'])],
                plot_params['yvargroup'],
                plot_params['yvar'],
                list_vars,
                flag_add_none = False,
                dicts_rename = {
                    'muse': st.session_state.dicts['muse']['ind_to_name']
                }
            )
            if sel_var != []:
                plot_params['yvargroup'] = sel_var[0]
                plot_params['yvar'] = sel_var[1]
                plot_params['roi_indices'] = utilmisc.get_roi_indices(
                    sel_var[1], 'muse'
                )

        elif tab == 'Filters':

            # Let user select sex var
            select_sex_var()

            # Let user pick an age range
            select_age_range()

        elif tab == 'Groups':

            # Select h var
            sel_var = utiluser.select_var_from_group(
                'Select group variable:',
                df_vars[df_vars.category.isin(['cat_vars'])],
                plot_params['hvargroup'],
                plot_params['hvar'],
                list_vars,
                flag_add_none = True,
            )
            if sel_var:
                plot_params['hvargroup'] = sel_var[0]
                plot_params['hvar'] = sel_var[1]

        elif tab == 'Fit':
            user_select_trend(plot_params)

        elif tab == 'Centiles':
            user_select_centiles(plot_params)

        elif tab == 'Plot Settings':
            user_select_plot_settings(plot_params)

        #elif tab == 'Add/Delete Plots':
            #user_add_plots(plot_params)

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
        
def panel_set_params_centile_plot(plot_params, var_groups_data, pipeline, list_vars, flag_hide_settings = False):
    """
    Panel to select centile plot args from the user
    """    
    plot_params['method'] = pipeline
    plot_params['flag_norm_centiles'] = False    

    if st.session_state.plot_settings['flag_hide_settings'] == 'Hide':
        return

    # Add tabs for parameter settings
    #with st.container(border=True):
    with st.sidebar:    
        tab = sac.tabs(
            items=[
                sac.TabsItem(label='Data'),
                sac.TabsItem(label='Centiles'),
                sac.TabsItem(label='Plot Settings'),
                #sac.TabsItem(label='Add/Delete Plots')
            ],
            size='sm',
            align='left'
        )
        df_vars = st.session_state.dicts['df_var_groups']
        if tab == 'Data':
            # Select x var
            sel_var = utiluser.select_var_from_group(
                'Select x variable:',
                df_vars[df_vars.group.isin(['demog'])],
                plot_params['xvargroup'],
                plot_params['xvar'],
                list_vars,
                flag_add_none = False,
                dicts_rename = {
                    'muse': st.session_state.dicts['muse']['ind_to_name']
                }
            )
            plot_params['xvargroup'] = sel_var[0]
            plot_params['xvar'] = sel_var[1]

            # Select y var
            sel_var = utiluser.select_var_from_group(
                'Select y variable:',
                df_vars[df_vars.category.isin(['roi'])],
                plot_params['yvargroup'],
                plot_params['yvar'],
                list_vars,
                flag_add_none = False,
                dicts_rename = {
                    'muse': st.session_state.dicts['muse']['ind_to_name']
                }
            )
                
            if sel_var != []:                
                plot_params['yvargroup'] = sel_var[0]
                plot_params['yvar'] = sel_var[1]
                plot_params['roi_indices'] = utilmisc.get_roi_indices(
                    sel_var[1], 'muse'
                )

        elif tab == 'Centiles':
            user_select_centiles(plot_params)

        elif tab == 'Plot Settings':
            user_select_plot_settings(plot_params)

        #elif tab == 'Add/Delete Plots':
            #user_add_plots(plot_params)

        # Set plot type
        plot_params['plot_type'] = 'scatter'
        
        # Set plot traces
        plot_params['traces'] = ['data']

        if plot_params['centile_values'] is not None:
            plot_params['traces'] = plot_params['traces'] + plot_params['centile_values']
        

def panel_show_plots():
    '''
    Panel to show plots
    '''
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if st.session_state.plots.loc[tmp_ind, 'flag_sel']:
            st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params.copy()

    # # Add a single plot if there is none
    # if st.session_state.plots.shape[0] == 0:
    #     st.session_state.plots = add_plot(
    #         st.session_state.plots, st.session_state.plot_params
    #     )

    # Show plots
    show_plots(
        st.session_state.plot_data['df_data'],
        st.session_state.plots,
        st.session_state.plot_settings
    )

    if st.session_state.sel_mrid is not None:
        show_mri()

def panel_show_centile_plots():
    '''
    Panel to show centile plots
    '''
    ## Update selected plots
    for tmp_ind in st.session_state.plots.index.tolist():
        if st.session_state.plots.loc[tmp_ind, 'flag_sel']:
            st.session_state.plots.at[tmp_ind, 'params'] = st.session_state.plot_params.copy()

    # # Add a single plot if there is none
    # if st.session_state.plots.shape[0] == 0:
    #     st.session_state.plots = add_plot(
    #         st.session_state.plots, st.session_state.plot_params
    #     )

    # Show plots
    show_plots(
        None,
        st.session_state.plots,
        st.session_state.plot_settings
    )






