import os
from time import sleep, time
from typing import Any, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import gui.utils_traces as utiltr
import gui.utils_mriview as utilmri
import gui.utils_widgets as utilwd
import gui.utils_dataprep as utildataprep
import gui.utils_reports as utilrep

from utils.utils_logger import setup_logger
logger = setup_logger()

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)  # or use a large number like 500

def get_index_in_list(in_list: list, in_item: str) -> Optional[int]:
    """
    Returns the index of the item in list, or None if item not found
    """
    if in_item not in in_list:
        return None
    else:
        return list(in_list).index(in_item)

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

def select_mrid(ptype, plot_id):
    '''
    Panel to select MRID for the plot
    '''
    ss = st.session_state
    pdict = ss[ptype]

    df = pdict['data'].get('df_user', None)
    if df is not None and 'MRID' in df.columns:
        utilwd.my_selectbox(
            f'{ptype}.selected.mrid', df['MRID'].tolist(), 'Subject'
        )


def filter_data(df, sel_params):
    df_filt = df.copy()  ## FIXME temp fix for filtering 
    return df_filt
    
    # if df is None:
    #     df_filt = None
    # else:
    #     df_filt = df.copy()
    #     if 'Sex' in df:
    #         df_filt = df_filt[df_filt.Sex.isin(sel_params.get('filter_sex', []))]
    #     if 'Age' in df:
    #         age_range = sel_params.get('filter_age', [0, 100])
    #         df_filt = df_filt[(df_filt.Age >= age_range[0]) & (df_filt.Age <= age_range[1])]

def select_filters(ptype, plot_id):
    '''
    Panel to select filters
    '''
    ss = st.session_state
    pdict = ss[ptype]

    with st.container(border=True):
        utilwd.my_multiselect(
            f'{ptype}.plots.{plot_id}.filter_sex', ['F','M'], 'Sex'     
        )

def select_min_max(ptype, plot_id):
    '''
    Panel to select layers
    '''
    def reset_value():

        update_plot_params(ptype, pdict, plot_id)

        for v in ['xmin', 'xmax', 'ymin', 'ymax']:
            vname = f'{ptype}.plots.{plot_id}.{v}'
            my_key = '_' + vname.replace('.', '_')
            st.session_state[my_key] = utilwd.get_nested_value(st.session_state, vname) 


    with st.container(horizontal=True):

        vname = f'{ptype}.plots.{plot_id}.xmin'
        utilwd.my_number_input(vname, 'X Min')

        vname = f'{ptype}.plots.{plot_id}.xmax'
        utilwd.my_number_input(vname, 'X Max')
        
    with st.container(horizontal=True):

        vname = f'{ptype}.plots.{plot_id}.ymin'
        utilwd.my_number_input(vname, 'Y Min')

        vname = f'{ptype}.plots.{plot_id}.ymax'
        utilwd.my_number_input(vname, 'Y Max')

    with st.container(horizontal=True):
        pdict = st.session_state[ptype]
        st.button(
            'Reset', 
            key = f'_btn_reset_{ptype}_{plot_id}',
            on_click = reset_value
        )
            


def select_centile_type(ptype):
    '''
    User panel to select centile type
    '''
    ss = st.session_state
    pdict = ss[ptype]
    
    list_types = pdict['settings'].get('cent_types', [])
    vname = f"{ptype}.data.cent_type"
    utilwd.my_selectbox(
        vname, list_types, 'Centile Type'
    )
    cent_type = pdict['data'].get('cent_type', None)

    if cent_type is None:
        return
    
    # Load centile data file if it's updated
    if utildataprep.load_centile_data(ptype) == 1:
        st.toast('Centile file loaded!')

def select_centile_values(ptype, plot_id):
    '''
    User panel to select centile values
    '''
    ss = st.session_state
    pdict = ss[ptype]
    
    list_centiles = pdict['settings'].get('cent_trace_types', [])
    vname = f"{ptype}.plots.{plot_id}.cent_values"
    utilwd.my_multiselect(
        vname, list_centiles, 'Centile Values'
    )

def select_trends(ptype,plot_id):
    '''
    Select trend type
    '''
    ss = st.session_state
    pdict = ss[ptype]

    list_trends = pdict['settings'].get('trend_types', [])
    vname = f"{ptype}.plots.{plot_id}.trend"
    my_key = '_' + vname.replace('.', '_')

    utilwd.my_selectbox(
        vname, list_trends, 'Trend'
    )
    sel_trend = st.session_state[my_key]

    if sel_trend is None or sel_trend == 'Select an option...':
        return

    if sel_trend == 'Linear':
        vname = f"{ptype}.plots.{plot_id}.flag_show_conf"
        utilwd.my_checkbox(
            vname, 'Add confidence interval'
        )

    elif sel_trend == 'Smooth LOWESS Curve':
        vname = f"{ptype}.plots.{plot_id}.lowess_s"
        utilwd.my_slider(
            vname, 'Smoothness', 0.4, 1.0, 0.1
        )
        # logger.debug(f"Selected LOWESS smoothness: {sel_lowess_s}")

def select_layers(ptype,plot_id):
    '''
    Panel to select layers
    '''
    ss = st.session_state
    pdict = ss[ptype]

    select_centile_values(ptype, plot_id)

    if ptype == 'user_plots':
        select_trends(ptype, plot_id)


def set_bounds(vmin, vmax, margin=0.1):
    '''
    Return (lower, upper) plot bounds with a margin added around [vmin, vmax].
    Padding is the larger of: a fraction of the range, or a fraction of the
    data magnitude — so near-identical or single-point data still gets a
    visible range proportional to the scale of the values.
    '''
    dx = vmax - vmin
    scale = max(abs(vmin), abs(vmax))
    pad = max(dx * margin, scale * margin * 0.5)
    if pad == 0:
        pad = 1.0
    return vmin - pad, vmax + pad

def update_plot_params(ptype, pdict, pl_id):
    '''
    Set data dependent parameters for the new plot
    '''
    curr_pl = pdict['plots'][pl_id]

    # Set selected yvar
    pdict['selected']['xlab'] = curr_pl['xlab']
    pdict['selected']['ylab'] = curr_pl['ylab']

    # Set bounds for variables
    if ptype == 'user_plots':
        df = pdict['data'].get('df_user', None)
        if df is None or df.shape[0] == 0:
            return   
        xmin = df[curr_pl['xvar']].min()
        xmax = df[curr_pl['xvar']].max()
        ymin = df[curr_pl['yvar']].min()
        ymax = df[curr_pl['yvar']].max()

    elif ptype == 'ref_plots':
        df = pdict['data'].get('df_cent', None)

        # st.write(df[df.VarName == curr_pl['yvar']])
        # time.sleep(10)

        if df is None or df.shape[0] == 0:
            return
        xmin = df['Age'].min()
        xmax = df['Age'].max()
        df_sel = df[df.VarName == curr_pl['yvar']].filter(regex='centile_') 
        ymin = df_sel.values.min()
        ymax = df_sel.values.max()

    (xmin, xmax) = set_bounds(xmin, xmax)
    curr_pl['xmin'] = xmin
    curr_pl['xmax'] = xmax
    (ymin, ymax) = set_bounds(ymin, ymax)
    curr_pl['ymin'] = ymin
    curr_pl['ymax'] = ymax

def add_plot(ptype: str):
    """
    Adds a new plot (new row to the plots dataframe)
    """
    pdict = st.session_state[ptype]

    # Add plot
    pl_id = 'plot_' + str(pdict.get('index', 0))
    pdict['index'] += 1
    pdict['df_plots'].loc[pl_id] = {'flag_sel': True}
    
    # Init parameters for the new plot
    pdict['plots'].update(
        {pl_id: pdict['params'].copy()}
    )
    
    # Set data dependent parameters for the new plot
    update_plot_params(ptype, pdict, pl_id)


def delete_sel_plots(ptype):
    """
    Removes plots selected by the user
    """
    pdict = st.session_state[ptype]

    # Update selected list of plots
    for pl_id in pdict['df_plots'].index.tolist():
        flag_sel = pdict['plots'][pl_id].get('flag_sel', False)
        pdict['df_plots'].loc[pl_id, 'flag_sel'] = flag_sel

    # Detect selected plots
    df = pdict['df_plots']
    list_sel = df[df.flag_sel == True].index.tolist()
    
    pdict['df_plots'] = df[df.flag_sel == False]
    for pl_id in list_sel:
        if pl_id in pdict['plots']:
            del pdict['plots'][pl_id]

    st.toast(f"Selected plots deleted: {list_sel}")

def delete_all_plots(ptype):
    """
    Removes all plots
    """
    """
    Removes plots selected by the user
    """
    pdict = st.session_state[ptype]

    list_sel = pdict['df_plots'].index.tolist()    
    pdict['df_plots'] = pd.DataFrame(columns=['flag_sel'])
    pdict['plots'] = {}
    
    # # Delete plot parameters from session state    
    # for pl_id in list_sel:
    #     dkeys = [k for k in st.session_state if k.startswith(f'{pl_id}_')]
    #     for key in dkeys:
    #         del st.session_state[key]
    # # Delete selected plots
    # st.session_state.df_plots = st.session_state.df_plots.drop(list_sel)

    st.toast(f"All plots deleted: {list_sel}")

def display_dist_plot(df, plot_params, plot_id, plot_settings):
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
        bin_size = x_range / plot_params.get('binnum', 10)
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

def display_scatter_plot(ptype, df, df_cent, plot_settings, plot_params, plot_id):
    '''
    Display scatter plot
    '''
    ss = st.session_state
    pdict = ss[ptype]

    logger.debug("    Function: display_scatter_plot")

    def callback_plot_clicked() -> None:
        """
        Set the active plot id to plot that was clicked
        """
        st.session_state.plots_active = plot_id

        # Detect x and y coor of the selected point 
        sel_info = st.session_state[f"_bubble_{ptype}_{plot_id}"]
        try:
            spoint = sel_info["selection"]["points"][0]
            pdict['selected']['xcoor'] = spoint['x']
            pdict['selected']['ycoor'] = spoint['y']
            st.toast(f"Selected point: {spoint['x']}, {spoint['y']}")
        except:
            return
 
        # Detect xvar and yvar info if a data point is selected
        try:
            cnum = sel_info["selection"]["points"][0].get('curve_number', None)
            if cnum is None or cnum != 0:
                return
            sind = sel_info["selection"]["point_indices"][0]

            # Detect MRID from the click info and save to session_state
            hvar = pdict['plots'][plot_id]['hvar']
            hind = get_index_in_list(df.columns.tolist(), hvar)
            if hind is None:
                sel_mrid = df.iloc[sind]["MRID"]
            else:
                if 'legendgroup' in sel_info["selection"]["points"][0]:
                    lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                    sel_mrid = df[df[hvar] == lgroup].iloc[sind]["MRID"]
                else:
                    sel_mrid = df.iloc[sind]["MRID"]

            ylab = pdict['plots'][plot_id]['ylab']
            pdict['selected']['mrid'] = sel_mrid
            pdict['selected']['ylab'] = ylab
            st.toast(f'Selected {sel_mrid}, variable {ylab}')    
        except:
            return

    ss = st.session_state

    # Set figure
    layout = go.Layout(
        height = plot_settings['h_init'] * plot_settings['h_coeff'],
        margin = dict(
            l = plot_settings['margin'],
            r = plot_settings['margin'],
            t = plot_settings['margin'],
            b = plot_settings['margin']
        ),
    )

    fig = go.Figure(layout=layout)

    # st.toast(pdict['settings'])
    utiltr.add_lines(
        pdict['settings']['flag_xline'],
        pdict['settings']['flag_yline'],
        pdict['selected']['xcoor'],
        pdict['selected']['ycoor'],
        fig
    )

    # Add traces
    traces = plot_params['cent_values']
    if ptype == 'user_plots':
        traces += ['data']
        if plot_params['trend'] == 'Linear':
            traces += ['lin_fit']
        elif plot_params['trend'] == 'Smooth LOWESS Curve':
            traces += ['lowess']
        if plot_params['flag_show_conf']:
            traces += ['conf_95%']
    plot_params['traces'] = traces

    # Filter centiles
    if df_cent is None:
        df_cent_filt = None
    else:
        df_cent_filt = df_cent.copy()
        if 'Age' in df_cent:
            df_cent_filt = df_cent_filt[(df_cent_filt.Age >= plot_params['filter_age'][0]) & (df_cent_filt.Age <= plot_params['filter_age'][1])]

    # Check centval flag and update variable names if needed
    if plot_params.get('flag_centval', False):
        plot_params['yvar'] = plot_params['yvar'] + '_cent'

    # Add axis labels
    fig.update_layout(
        xaxis_title=plot_params["xlab"], yaxis_title=plot_params["ylab"]
    )

    cmaps = plot_settings['cmaps']
    alphas = plot_settings['alphas']
    flag_show_legend = plot_settings['flag_show_legend']
    w_fit = plot_settings['w_fit']
    w_cent = plot_settings['w_cent']
    cent_trace_types = plot_settings['cent_trace_types']

    # Add data scatter
    if df is not None:
        utiltr.add_trace_scatter(df, plot_params, cmaps, alphas, flag_show_legend, fig)

    # Add linear fit
    trend = plot_params.get('trend', '')
    if trend == 'Linear':
        if df is not None:
            utiltr.add_trace_linreg(df, plot_params, cmaps, alphas, w_fit, flag_show_legend, fig)

    # Add non-linear fit
    if trend == 'Smooth LOWESS Curve':
        if df is not None:
            utiltr.add_trace_lowess(df, plot_params, cmaps, alphas, w_fit, flag_show_legend, fig)

    # Add centile trace
    if df_cent_filt is not None:
        utiltr.add_trace_centile(
            df_cent_filt, plot_params, cmaps, alphas,
            w_cent, flag_show_legend, cent_trace_types, fig
        )

    # Add selected dot
    if df is not None:
        sel_mrid = pdict['selected']['mrid']
        if sel_mrid is not None:
            utiltr.add_trace_dot(df, sel_mrid, plot_params, flag_show_legend, fig)

    st.plotly_chart(
        fig, key=f"_bubble_{ptype}_{plot_id}", on_select=callback_plot_clicked
    )

    return fig

def show_plots(ptype):
    """
    Display data plots
    """
    logger.debug("--- Function: show_plots ---")

    ss = st.session_state
    pdict = ss[ptype]

    df_plots = pdict['df_plots']
    df = pdict['data'].get('df_user', None)

    df_cent = pdict['data'].get('df_cent', None)
    
    # Read plot ids
    list_plots = df_plots.index.tolist()
    logger.debug(f"List of plots: {list_plots}")
        
    # Set number of plots in each row
    num_plot = len(list_plots)
    if pdict['settings'].get('flag_resize', True):
        plots_per_row = int(np.min([num_plot, pdict['settings'].get('num_per_row', 1)]))
    else:
        plots_per_row = pdict['settings'].get('num_per_row', 1)

    logger.debug(f"Number of plots: {num_plot}")

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
    for i, plot_id in enumerate(list_plots):        
        column_no = i % plots_per_row
        if column_no == 0:
            blocks = st.columns(plots_per_row)

        logger.debug(f'plot_id: {plot_id}, column_no: {column_no}')
        with blocks[column_no]:
            # Show plot parameters menu
            set_plot_params(ptype, plot_id)

            # Read plot parameters
            plot_settings = pdict.get('settings', {})
            plot_params = pdict['plots'].get(plot_id, {})
            # logger.debug(f'  sel params: {plot_params}')

            # Filter data
            df_filt = None
            if df is not None:
                df_filt = filter_data(df, plot_params)

            # Show plot based on the selected parameters
            if plot_params.get('plot_type') == "dist": 
                new_plot = display_dist_plot(
                    df_filt, plot_params, plot_id, plot_settings
                )
                
            elif plot_params.get('plot_type') == "scatter": 
                new_plot = display_scatter_plot(
                    ptype, df_filt, df_cent, plot_settings, plot_params, plot_id
                )
            else:
                st.warning(f"Plot type {plot_params.get('plot_type')} not supported!")
                new_plot = None

            if new_plot is not None:
                pdict.setdefault('_figs', {})[plot_id] = new_plot

def panel_show_plots(ptype):
    '''
    Panel to show plots
    '''
    pdict = st.session_state[ptype]

    df_plots = pdict['df_plots']

    # Check plots
    if df_plots is None or df_plots.shape[0] == 0:
        # st.toast('No plots to show. Please add a plot using the Plots tab.')
        return

    horizontal = pdict['settings'].get('mri_layout_horizontal', True)
    flag_mri = pdict['settings'].get('mri_flag_show', True)

    if flag_mri:
        if horizontal:
            col_plots, col_mri = st.columns([2, 1])
            with col_plots:
                st.markdown('`Plots`')
                show_plots(ptype)
            with col_mri:
                st.markdown('`Image Viewer`')
                utilmri.panel_view_seg(ptype)
        else:
            st.markdown('`Plots`')
            show_plots(ptype)
            st.markdown('`Image Viewer`')
            utilmri.panel_view_seg(ptype)

    else:
        st.markdown('`Plots`')
        show_plots(ptype)

def _group_vars_in_data(group_info, data_vars, roi_lists, sel_sublist=None):
    """
    Return [(col_name, display_name), ...] for a var_group entry,
    restricted to variables actually present in data_vars.

    Groups with a 'prefix' key are ROI groups — col names are built as
    prefix + roi_index drawn from roi_lists.  sel_sublist (a roi_list key)
    restricts to one sub-list; None returns all indices across all lists
    whose atlas matches the prefix.

    Groups with a 'values' key list variable names directly.
    """
    prefix = group_info.get('prefix')
    values = group_info.get('values', [])

    if prefix:
        atlas = 'muse' if 'muse' in prefix.lower() else None

        if sel_sublist and sel_sublist in roi_lists:
            indices = [str(v) for v in roi_lists[sel_sublist].get('values', [])]
        else:
            seen = {}
            for linfo in roi_lists.values():
                if atlas is None or linfo.get('atlas') == atlas:
                    for v in linfo.get('values', []):
                        seen[str(v)] = None
            indices = list(seen)

        col_names = [f"{prefix}{idx}" for idx in indices]
        return [(c, data_vars.dict_fwd.get(c, c)) for c in col_names if c in data_vars.dict_fwd]
    else:
        return [(v, data_vars.dict_fwd.get(v, v)) for v in values if v in data_vars.dict_fwd]

@st.dialog('Browse Variable')
def _var_browser_dialog(vname, vlabel, vtext, col_dict, var_groups, roi_lists):
    '''
    Modal dialog for browsing and selecting a variable by category / sub-list.
    '''
    ss = st.session_state

    st.markdown(f'**{vtext}**')

    my_key = '_' + vname.replace('.', '_')

    # ── Step 1: category ──────────────────────────────────────────────
    group_keys   = list(var_groups.keys())
    group_labels = {k: var_groups[k].get('label', k) for k in group_keys}

    sel_group = st.selectbox(
        'Category',
        group_keys,
        format_func=lambda k: group_labels[k],
        index=None,
        placeholder='Select a category…',
        key=f'_{my_key}_group',
    )
    if sel_group is None:
        return

    ginfo  = var_groups[sel_group]
    prefix = ginfo.get('prefix')

    # ── Step 2 (ROI groups only): sub-list ────────────────────────────
    sel_sublist = None
    if prefix:
        atlas = 'muse' if 'muse' in prefix.lower() else None
        avail_lists = {
            k: v for k, v in roi_lists.items()
            if atlas is None or v.get('atlas') == atlas
        }
        if avail_lists:
            sel_sublist = st.selectbox(
                'ROI list',
                list(avail_lists.keys()),
                format_func=lambda k: (
                    f"{avail_lists[k].get('desc', k)}"
                    f" ({len(avail_lists[k].get('values', []))})"
                ),
                index=None,
                placeholder='Select an ROI list…',
                key=f'_vb_{vname}_sublist',
            )
            if sel_sublist is None:
                return

    # ── Step 3: variable ──────────────────────────────────────────────
    var_pairs = _group_vars_in_data(ginfo, col_dict, roi_lists, sel_sublist)
    if not var_pairs:
        st.caption(':orange[No variables from this group found in the loaded data.]')
        return

    display_names = [d for _, d in var_pairs]
    sel_display = st.selectbox(
        'Variable',
        display_names,
        index=None,
        placeholder='Select a variable…',
        key=f'_vb_{my_key}_var',
    )

    # ── Select button ─────────────────────────────────────────────────
    if sel_display:
        if st.button('Select', key=f'_vb_{my_key}_select', type='primary'):
            
            utilwd.set_nested_value(ss, vname, sel_display)
            ss[my_key] = sel_display

            st.rerun()  # closes the dialog


def _var_selectbox(ptype, pl_id, col_dict, vname, vlabel, vtext):

    ss = st.session_state
    all_labels = ['Age'] + col_dict.df_cols['renamed'].tolist()

    vlabel_full = f'{ptype}.plots.{pl_id}.{vlabel}'

    var_groups = ss.dicts.get('var_groups', {})
    roi_lists  = ss.dicts.get('roi_lists', {})

    col_btn, col_sel = st.columns([1, 5], vertical_alignment='bottom')
    with col_btn:
        if st.button(':material/manage_search:', key=f'_vb_{vlabel_full}_open', 
        use_container_width=True):
            
            # Select variable for the axis    
            _var_browser_dialog(
                vlabel_full, vlabel, vtext, col_dict, var_groups, roi_lists
            )

            # Update corresponding variable name in session state
            sel_label = ss[ptype]['plots'][pl_id].get(vlabel, None)
            sel_var = col_dict.dict_rev.get(sel_label, None)
            if sel_var is not None:
                ss[ptype]['plots'][pl_id][vname] = sel_var


    with col_sel:
        # Select variable for the axis
        utilwd.my_selectbox(
            vlabel_full, all_labels, vtext
        )

        # Update corresponding variable name in session state
        sel_label = ss[ptype]['plots'][pl_id].get(vlabel, None)
        sel_var = col_dict.dict_rev.get(sel_label, None)
        if sel_var is not None:
            ss[ptype]['plots'][pl_id][vname] = sel_var


def set_plot_params(ptype, plot_id):
    """
    Panel for selecting plotting parameters, with pipeline as first tab.
    """
    ss = st.session_state
    pdict = ss[ptype]

    data_vars = pdict['data'].get('vars', None)

    if ptype == 'ref_plots':
        sel_tabs = ['tab0', 'tab1', 'tab2', 'tab3', 'tab6']
        tab0, tab1, tab2, tab3, tab6 = st.tabs(
            [":material/visibility_off:", "Variables", "Values", "Layers", "Select"],
            on_change='rerun',
            key = f"_tabs_plot_params_{plot_id}"
        )
    else:
        sel_tabs = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6']
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [":material/visibility_off:", "Variables", "Values", "Layers", "Filters", "Subjects", "Select"],
            on_change='rerun',
            key = f"_tabs_plot_params_{plot_id}"
        )

    if 'tab1' in sel_tabs and tab1.open:
        with tab1:
            with st.container(border=True):
                
                col_dict = pdict['data']['col_dict']

                # Variable X:
                _var_selectbox(
                    ptype, plot_id, col_dict, 'xvar', 'xlab', 'Variable X'
                )

                # Variable Y:
                _var_selectbox(
                    ptype, plot_id, col_dict, 'yvar', 'ylab', 'Variable Y'
                )

                update_plot_params(ptype, pdict, plot_id)

    #             # Show ICV-normalized variable if available
    #             utilwd.my_checkbox(f'{plot_id}_flag_icvcorr', 'Correct for ICV')

    #             # Show centile value if available
    #             utilwd.my_checkbox(f'{plot_id}_flag_centval', 'Show Centile Values')


    if 'tab2' in sel_tabs and tab2.open:
        with tab2:
            with st.container(border=True):
                select_min_max(ptype, plot_id)


    if 'tab3' in sel_tabs and tab3.open:
        with tab3:
            with st.container(border=True):
                select_layers(ptype, plot_id)


    if 'tab4' in sel_tabs and tab4.open:
        with tab4:
            with st.container(border=True):
                select_filters(ptype, plot_id)

    if 'tab5' in sel_tabs and tab5.open:
        with tab5:
            with st.container(border=True):
                select_mrid(ptype, plot_id)

    if 'tab6' in sel_tabs and tab6.open:
        with tab6:
            with st.container(border=True):
                vname = f"{ptype}.plots.{plot_id}.flag_sel"
                utilwd.my_checkbox(vname, 'Select')
    

def plot_add_delete(ptype):
    '''
    Panel to add/delete plots
    '''
    ss = st.session_state
    pdict = ss[ptype]

    with st.container(horizontal=True, horizontal_alignment="left"):
        
        # Preconf setting with multiple plots
        if st.button(
            '', icon=':material/auto_fix_high:', key=f'_{ptype}_add_auto',
            help = 'Auto Plot'
        ):
            col_dict = pdict['data']['col_dict']

            # Set number of plot columns
            num_per_row = pdict['auto'].get('num_per_row', 1)
            pdict['settings']['num_per_row'] = num_per_row

            # Add plots
            list_labels = pdict['auto'].get('list_labels', [])
            for i in range(len(list_labels)):
                # Set label and var name for y axis using the list of labels provided in the auto settings
                pdict['params']['ylab'] = list_labels[i]
                sel_var = col_dict.dict_rev.get(list_labels[i], None)
                logger.debug(f"Auto-adding plot with ylab: {pdict['params']['ylab']}, sel_var: {sel_var}")
                if sel_var is not None:
                    pdict['params']['yvar'] = sel_var

                # Add plot
                add_plot(ptype)

        if st.button('Add', key='plot_controls_add_plot'):
            if st.session_state.flag_user_data:
                if pdict['data']['df_user'] is not None:
                    add_plot(ptype)
                else:
                    st.toast('No data to plot. Please prepare data first.')
            else:
                add_plot(ptype)

        if pdict['df_plots'].shape[0] == 0:
            return

        if st.button('Delete Selected', key='plot_controls_delete_selected'):
            delete_sel_plots(ptype)

        if st.button('Delete All', key='plot_controls_delete_all'):
            delete_all_plots(ptype)


def plot_select_settings(ptype):
    '''
    Panel to select plot settings
    '''
    ss = st.session_state
    pdict = ss[ptype]

    tab1, tab2, tab3, tab4 = st.tabs(
            ["Centiles", "Layout", "Components", "Auto Plot"],
            on_change='rerun',
            key = f"_tabs_plot_settings_{ptype}"
        )

    if tab1.open:
        with tab1:
            with st.container(horizontal=True, border=False):
                select_centile_type(ptype)

    if tab2.open:
        with tab2:
            with st.container(horizontal=True, border=False):
                utilwd.my_slider(
                    f'{ptype}.settings.num_per_row', 'Number of plots per row',
                    pdict['settings'].get('min_per_row', 1),
                    pdict['settings'].get('max_per_row', 10),
                    1
                )
                utilwd.my_checkbox(
                    f'{ptype}.settings.flag_resize', 'Use Full Width'
                )
            with st.container(horizontal=True, border=False):
                utilwd.my_slider(
                    f'{ptype}.settings.h_coeff', "Plot Height",
                    pdict['settings'].get('h_coeff_min', 0.1),
                    pdict['settings'].get('h_coeff_max', 1.0),
                    pdict['settings'].get('h_coeff_step', 0.1)
                )

    if tab3.open:
        with tab3:
            # Checkbox to show/hide plot legend
            with st.container(horizontal=True, border=False):
                utilwd.my_checkbox(
                    f'{ptype}.settings.flag_show_legend', 'Show Legend'
                )
                utilwd.my_checkbox(
                    f'{ptype}.settings.flag_xline', 'Show Horizontal Guide'
                )
                utilwd.my_checkbox(
                    f'{ptype}.settings.flag_yline', 'Show Vertical Guide'
                )
                
            with st.container(horizontal=True, border=False):
                utilwd.my_checkbox(
                    f'{ptype}.settings.mri_flag_show', 'Show Image Panel'
                )
                if pdict['settings']['mri_flag_show']:
                    utilwd.my_checkbox(
                        f'{ptype}.settings.mri_layout_horizontal', 'Show Side-by-Side'
                    )

    if tab4.open:
        with tab4:
            with st.container(horizontal=True, border=False):
        # Select auto variables
                st.pills('Selected variables', pdict['auto'].get('list_labels', []), selection_mode="single")

                if st.checkbox('Select Auto Variables', key=f'{ptype}_flag_select_vars'):
                    df_cent = pdict['data'].get('df_cent', None)
                    col_dict = pdict['data']['col_dict']
                    list_vars = col_dict.df_cols['renamed'].tolist()
                    sel_vars = st.pills(
                        'Auto-plot variables',
                        list_vars,
                        selection_mode = 'multi',
                        default = pdict['auto'].get('list_labels', []),
                        key = f'{ptype}_pills_auto_sel_vars'
                    )
                    if st.button(
                        'Select',
                        key=f'{ptype}_button_auto_sel_vars'
                    ):
                        pdict['auto']['list_labels'] = sel_vars
                    



def plot_set_controls(ptype):
    '''
    Show controls to add plot and select settings
    '''
    _titles = {'ref_plots': 'NiChart Reference Charts', 'user_plots': 'NiChart My Charts'}

    tab0, tab1, tab2, tab3 = st.tabs(
        [':material/visibility_off:', 'Plots', 'Settings', 'Save Report'],
        on_change='rerun',
    )

    if tab1.open:
        with tab1:
            with st.container(border=True, horizontal=True, horizontal_alignment="center", width='stretch'):
                plot_add_delete(ptype)

    if tab2.open:
        with tab2:
            with st.container(border=True):
                plot_select_settings(ptype)

    if tab3.open:
        with tab3:
            utilrep.panel_save_plots(ptype, _titles.get(ptype, 'NiChart Charts'))

def panel_view_vars():
    """
    View variables
    """
    logger.debug('    Function: panel_view_vars')

    ss= st.session_state

    # Set plot type based on the data type (user vs reference)
    if ss.flag_user_data:
        ptype='user_plots'
    else:
        ptype='ref_plots'

    ## Plot controls to add plot
    plot_set_controls(ptype)

    ## Show plots
    panel_show_plots(ptype)




