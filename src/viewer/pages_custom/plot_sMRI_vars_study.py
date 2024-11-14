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
                st.session_state.flags['csv_plot'],
                "visible"
            )
        else:  # st.session_state.app_type == 'desktop'
            st.session_state.is_updated['csv_plot'] = utilst.util_select_file(
                "selected_data_file",
                "Data csv",
                st.session_state.paths["csv_plot"],
                st.session_state.paths["last_in_dir"],
                st.session_state.flags['csv_plot']
            )

        if os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")
            st.session_state.flags['csv_plot'] = True

        # Read input csv
        if st.session_state.is_updated['csv_plot']:
            st.session_state.plot_var['df_data'] = utildf.read_dataframe(st.session_state.paths["csv_plot"])
            st.session_state.is_updated['csv_plot'] = False

# Panel for renaming variables
icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
show_panel_rename = st.checkbox(
    f":material/new_label: Rename Variables (optional)",
    disabled = not st.session_state.flags['csv_plot'],
    value = False
)
if show_panel_rename:
    with st.container(border=True):

        msg_help = 'Rename numeric ROI indices to ROI names. \n\n If a dictionary is not provided for your data type, please continue with the next step!'

        sel_dict = st.selectbox(
            'Select ROI dictionary',
            st.session_state.rois['roi_dict_options'],
            index=None,
            help = msg_help
        )
        if sel_dict is not None:
            st.session_state.rois['sel_roi_dict'] = sel_dict
            ssroi = st.session_state.rois
            df_tmp = pd.read_csv(
                os.path.join(ssroi['path'], ssroi['roi_csvs'][ssroi['sel_roi_dict']])
            )
            dict1 = dict(zip(df_tmp["Index"].astype(str), df_tmp["Name"].astype(str)))
            dict2 = dict(zip(df_tmp["Name"].astype(str), df_tmp["Index"].astype(str)))
            st.session_state.rois['roi_dict'] = dict1
            st.session_state.rois['roi_dict_inv'] = dict2

            # Get a list of columns that match the dict key
            df = st.session_state.plot_var['df_data']
            df_tmp = df[df.columns[df.columns.isin(st.session_state.rois['roi_dict'].keys())]]
            df_tmp2 = df_tmp.rename(columns=st.session_state.rois['roi_dict'])
            if df_tmp.shape[1] == 0:
                st.warning('None of the variables were found in the dictionary!')
            else:
                with st.container(border=True):
                    tmp_cols = st.columns(3)
                    with tmp_cols[0]:
                        st.selectbox('Initial ...', df_tmp.columns, 0)
                    with tmp_cols[1]:
                        st.selectbox('Renamed to ...', df_tmp2.columns, 0)

                if st.button('Approve renaming'):
                    df = df.rename(columns=st.session_state.rois['roi_dict'])
                    st.session_state.plot_var['df_data'] = df
                    st.success(f'Variables are renamed!')

# Panel for selecting variables
icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
show_panel_select = st.checkbox(
    ":material/playlist_add: Select Variables (optional)",
    disabled = not st.session_state.flags['csv_plot'],
    value = False
)
if show_panel_select:
    with st.container(border=True):

        with open(st.session_state.dict_categories, 'r') as f:
            dict_categories = json.load(f)

        # User selects a category to include

        cols_tmp = st.columns((1,3,1), vertical_alignment="bottom")
        with cols_tmp[0]:
            sel_cat = st.selectbox('Select category', list(dict_categories.keys()), index = None)

        if sel_cat is None:
            sel_vars = []
        else:
            sel_vars = dict_categories[sel_cat]

        with cols_tmp[1]:
            sel_vars = st.multiselect('Which ones to keep?', sel_vars, sel_vars)

        with cols_tmp[2]:
            if st.button('Add selected variables ...'):
                sel_vars_uniq = [v for v in sel_vars if v not in st.session_state.plot_sel_vars]
                st.session_state.plot_sel_vars += sel_vars_uniq

        sel_vars_all = st.multiselect(
            'Add variables',
            st.session_state.plot_sel_vars,
            st.session_state.plot_sel_vars
        )

        if st.button('Select variables ...'):
            st.success(f'Selected variables: {sel_vars_all}')
            df = df[st.session_state.plot_sel_vars]

# Panel for filtering variables
icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
show_panel_filter = st.checkbox(
    ":material/filter_alt: Filter Data (optional)",
    disabled = not st.session_state.flags['csv_plot'],
    value = False
)
if show_panel_filter:
    df = st.session_state.plot_var['df_data']
    with st.container(border=True):
        df = utildf.filter_dataframe(df)


# Panel for displaying plots
show_panel_plots = st.checkbox(
    f":material/bid_landscape: Plot Data",
    disabled = not st.session_state.flags['csv_plot']
)

# with Profiler() as pr:

if show_panel_plots:
    if st.session_state.plot_var['df_data'].shape[0] == 0:
        st.session_state.plot_var['df_data'] = utildf.read_dataframe(st.session_state.paths["csv_plot"])
    df = st.session_state.plot_var['df_data']

    ################
    # Sidebar parameters
    with st.sidebar:
        # Button to add plot
        tmp_cols = st.columns((1,1), vertical_alignment='bottom')
        with tmp_cols[0]:
            plot_type = st.selectbox(
                "Plot Type",
                ["Scatter Plot", "Distribution Plot"],
                index = 0
            )
            if plot_type is not None:
                st.session_state.plot_var['plot_type'] = plot_type
        
        with tmp_cols[1]:
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
                plot_type = st.session_state.plots.loc[plot_ind, 'plot_type']
                if plot_type == 'Scatter Plot':
                    new_plot = utilpl.display_scatter_plot(
                        df,
                        plot_ind,
                        not st.session_state.plot_var['hide_settings'],
                        st.session_state.sel_mrid
                    )
                elif plot_type == 'Distribution Plot':
                    new_plot = utilpl.display_dist_plot(
                        df,
                        plot_ind,
                        not st.session_state.plot_var['hide_settings'],
                        st.session_state.sel_mrid
                    )                    
                plots_arr.append(new_plot)

    ################
    # Show images
    if st.session_state.plot_var['show_img']:
        # Check if data point selected
        if st.session_state.sel_mrid == "":
            st.warning("Please select a subject on the plot!")

        else:
            if not utilvi.check_image_underlay() or not utilvi.check_image_overlay():
                utilvi.get_image_paths()

        if not st.session_state.sel_mrid == "":
            with st.spinner("Wait for it..."):

                # Get indices for the selected var
                list_rois = utilroi.get_list_rois(
                    st.session_state.plots.loc[st.session_state.plot_active, "yvar"],
                    st.session_state.rois['roi_dict_inv'],
                    st.session_state.rois['roi_dict_derived'],
                )

                # Process image and mask to prepare final 3d matrix to display
                flag_files = 1
                if not os.path.exists(st.session_state.paths["sel_img"]):
                    flag_files = 0
                    warn_msg = (
                        f"Missing underlay image: {st.session_state.paths['sel_img']}"
                    )
                if not os.path.exists(st.session_state.paths["sel_seg"]):
                    flag_files = 0
                    warn_msg = (
                        f"Missing overlay image: {st.session_state.paths['sel_seg']}"
                    )

                if flag_files == 0:
                    st.warning(warn_msg)
                else:
                    img, mask, img_masked = utilni.prep_image_and_olay(
                        st.session_state.paths["sel_img"],
                        st.session_state.paths["sel_seg"],
                        list_rois,
                        crop_to_mask
                    )

                    # Detect mask bounds and center in each view
                    mask_bounds = utilni.detect_mask_bounds(mask)

                    # Show images
                    blocks = st.columns(len(list_orient))
                    for i, tmp_orient in enumerate(list_orient):
                        with blocks[i]:
                            ind_view = utilni.img_views.index(tmp_orient)
                            if not is_show_overlay:
                                utilst.show_img3D(
                                    img, ind_view, mask_bounds[ind_view, :], tmp_orient
                                )
                            else:
                                utilst.show_img3D(
                                    img_masked,
                                    ind_view,
                                    mask_bounds[ind_view, :],
                                    tmp_orient
                                )

# FIXME: For DEBUG
utilst.add_debug_panel()
