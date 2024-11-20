import os
import glob
import pandas as pd
import json
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_nifti as utilni
import utils.utils_trace as utiltr
import utils.utils_st as utilst
import utils.utils_dataframe as utildf
import utils.utils_viewimg as utilvi
import utils.utils_plot as utilpl
from stqdm import stqdm

# from wfork_streamlit_profiler import Profiler
# with Profiler():

st.markdown(
    """
    - Plot study data to visualize imaging variables
    - With options to:
    - Select target variables to plot
    - View reference distributions (centile values of the reference dataset)
    - Filter data
    - View MRI images and segmentations for selected data points
    """
)

st.divider()

# Panel for selecting the working dir
icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
show_panel_wdir = st.checkbox(
    f":material/folder_shared: Working Directory {icon}",
    value = False
)
if show_panel_wdir:
    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["dir_out"] = True

# Panel for selecting data csv
msg = st.session_state.app_config[st.session_state.app_type]['msg_infile']
icon = st.session_state.icon_thumb[st.session_state.flags['csv_plot']]
show_panel_incsv = st.checkbox(
    f":material/upload: {msg} Data {icon}",
    disabled = not st.session_state.flags['dir_out'],
    value = False
)
if show_panel_incsv:
    with st.container(border=True):
        if st.session_state.app_type == "CLOUD":
            st.session_state.is_updated['csv_plot'] = utilst.util_upload_file(
                st.session_state.paths["csv_plot"],
                "Input data csv file",
                "key_in_csv",
                False,
                "visible"
            )
        else:  # st.session_state.app_type == 'desktop'
            st.session_state.is_updated['csv_plot'] = utilst.util_select_file(
                "selected_data_file",
                "Data csv",
                st.session_state.paths["csv_plot"],
                st.session_state.paths["last_in_dir"],
                False
            )

        if os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")
            st.session_state.flags['csv_plot'] = True

        # Read input csv
        if st.session_state.is_updated['csv_plot']:
            st.session_state.plot_var['df_data'] = utildf.read_dataframe(st.session_state.paths["csv_plot"])
            st.session_state.is_updated['csv_plot'] = False

# Panel for displaying plots
show_panel_plots = st.checkbox(
    f":material/bid_landscape: Plot Data",
    disabled = not st.session_state.flags['csv_plot']
)
if show_panel_plots:
    if st.session_state.plot_var['df_data'].shape[0] == 0:
        st.session_state.plot_var['df_data'] = utildf.read_dataframe(st.session_state.paths["csv_plot"])
    df = st.session_state.plot_var['df_data']

    ################
    # Sidebar parameters
    with st.sidebar:
        # Slider to set number of plots in a row
        btn_plots = st.button("Add plot", disabled = False)

        st.session_state.plot_const['num_per_row'] = st.slider(
            "Plots per row",
            st.session_state.plot_const['min_per_row'],
            st.session_state.plot_const['max_per_row'],
            st.session_state.plot_const['num_per_row'],
            disabled = False
        )

        st.session_state.plot_h_coeff = st.slider(
            "Plot height",
            min_value=st.session_state.plot_const['h_coeff_min'],
            max_value=st.session_state.plot_const['h_coeff_max'],
            value=st.session_state.plot_const['h_coeff'],
            step=st.session_state.plot_const['h_coeff_step'],
            disabled = False
        )

        # Checkbox to show/hide plot options
        st.session_state.plot_var['hide_settings'] = st.checkbox(
            "Hide plot settings",
            value=st.session_state.plot_var['hide_settings'],
            disabled = False
        )

        # Checkbox to show/hide plot legend
        st.session_state.plot_var['hide_legend']= st.checkbox(
            "Hide legend",
            value=st.session_state.plot_var['hide_legend'],
            disabled = False
        )

        st.divider()

        # Checkbox to show/hide mri image
        st.session_state.plot_var['show_img']= st.checkbox(
            "Show image",
            value=st.session_state.plot_var['show_img'],
            disabled=False
        )

        # Selected id
        if st.session_state.sel_mrid != '':
            list_mrid = df.MRID.sort_values().tolist()
            sel_ind = list_mrid.index(st.session_state.sel_mrid)
            st.session_state.sel_mrid = st.selectbox(
                "Selected subject",
                list_mrid,
                sel_ind,
                help='Select a subject from the list, or by clicking on data points on the plots'
            )

        # Selected roi rois
        if st.session_state.sel_roi != '':
            list_roi = df.columns.sort_values().tolist()
            sel_ind = list_roi.index(st.session_state.sel_roi)
            st.session_state.sel_roi = st.selectbox(
                "Selected ROI",
                list_roi,
                sel_ind,
                help='Select an ROI from the list'
            )

        if st.session_state.plot_var['show_img']:
            # Show mrids

            # Create a list of checkbox options
            list_orient = st.multiselect(
                "Select viewing planes:",
                utilni.img_views,
                utilni.img_views
            )

            # View hide overlay
            is_show_overlay = st.checkbox(
                "Show overlay",
                True
            )

            # Crop to mask area
            crop_to_mask = st.checkbox(
                "Crop to mask",
                True
            )

    ################
    # Show plots

    # Add a plot (a first plot is added by default; others at button click)
    if st.session_state.plots.shape[0] == 0 or btn_plots:
        # Select xvar and yvar, if not set yet
        num_cols = df.select_dtypes(include='number').columns
        if num_cols.shape[0] > 0:
            if st.session_state.plot_var['xvar'] == '':
                st.session_state.plot_var['xvar'] = num_cols[0]
                if st.session_state.plot_var['yvar'] == '':
                    if num_cols.shape[0] > 1:
                        st.session_state.plot_var['yvar'] = num_cols[1]
                    else:
                        st.session_state.plot_var['yvar'] = num_cols[0]
            utilpl.add_plot()
        else:
            st.warning('No numeric columns in data!')

    # Read plot ids
    df_p = st.session_state.plots
    list_plots = df_p.index.tolist()
    plots_per_row = st.session_state.plot_const['num_per_row']

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot

    if df.shape[0] > 0:
        plots_arr = []

        # FIXME: this created a bug ???
        #for i, plot_ind in stqdm(
            #enumerate(list_plots), desc="Rendering plots ...", total=len(list_plots)
        #):
        for i, plot_ind in enumerate(list_plots):
            column_no = i % plots_per_row
            if column_no == 0:
                blocks = st.columns(plots_per_row)
            with blocks[column_no]:

                new_plot = utilpl.display_plot(
                    df,
                    plot_ind,
                    not st.session_state.plot_var['hide_settings'],
                    st.session_state.sel_mrid
                )
                plots_arr.append(new_plot)
