import os
import glob
import pandas as pd
import json
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_trace as utiltr
import utils.utils_st as utilst
import utils.utils_dataframe as utildf
import utils.utils_viewimg as utilvi
import utils.utils_plot as utilpl
from stqdm import stqdm

## Alias for session state 
#ss = st.session_state

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Select data csv from existing data files
flag_update = False
if not os.path.exists(st.session_state.paths["csv_plot"]):
    if os.path.exists(st.session_state.paths["csv_mlscores"]):
        st.session_state.paths["csv_plot"] = st.session_state.paths["csv_mlscores"]
        flag_update = True
    elif os.path.exists(st.session_state.paths["csv_dlmuse"]):
        st.session_state.paths["csv_plot"] = st.session_state.paths["csv_dlmuse"]
        flag_update = True
if flag_update:
    st.session_state.is_updated['csv_plot'] = True

# Panel for selecting input csv
flag_disabled = not st.session_state.flags['dset']
if st.session_state.app_type == "CLOUD":
    with st.expander(":material/upload: Upload Data", expanded=False):  # type:ignore
        flag_update = utilst.util_upload_file(
            st.session_state.paths["csv_plot"],
            "Input data csv file",
            "key_in_csv",
            flag_disabled,
            "visible",
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")

else:  # st.session_state.app_type == 'DESKTOP'
    with st.expander(":material/upload: Select Data", expanded=False):
        flag_update = utilst.util_select_file(
            "selected_data_file",
            "Data csv",
            st.session_state.paths["csv_plot"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")
if flag_update:
    st.session_state.is_updated['csv_plot'] = True

# Read input csv
df = st.session_state.df_plot
if df.shape[0] == 0 or st.session_state.is_updated['csv_plot']:
    df = utildf.read_dataframe(st.session_state.paths["csv_plot"])
    st.session_state.is_updated['csv_plot'] = False

print(st.session_state.paths["csv_plot"])
print(df.shape)


# Panel for renaming variables
flag_disabled = df.shape[0] == 0
with st.expander(":material/playlist_add: Rename Variables", expanded=False):  # type:ignore

    st.info('This is an optional step to rename the columns of the data file (example: column names are numeric roi indices)')

    sel_dict = st.selectbox('Select dictionary', ['muse_all'], None)
    if sel_dict is not None:
        st.session_state.paths["csv_roidict"] = st.session_state.dicts[sel_dict]

    if os.path.exists(st.session_state.paths["csv_roidict"]):

        df_dict = pd.read_csv(st.session_state.paths["csv_roidict"])
        
       
        dict_r1 = dict(
            zip(df_dict["Index"].astype(str), df_dict["Name"].astype(str))
        )
        dict_r2 = dict(
            zip(df_dict["Name"].astype(str), df_dict["Index"].astype(str))
        )
        st.session_state.roi_dict = dict_r1
        st.session_state.roi_dict_rev = dict_r2
    
    if st.session_state.roi_dict is not None:
        df = df.rename(columns=st.session_state.roi_dict)
        st.success(f'Renamed variables')

# Panel for selecting variables
flag_disabled = df.shape[0] == 0
with st.expander(":material/playlist_add: Select Variables", expanded=False):  # type:ignore
    st.info('This is an optional step. Use it to select a subset of variables, or just skip to continue with all variables')

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
flag_disabled = df.shape[0] == 0
with st.expander(":material/filter_alt: Filter Data", expanded=False):  # type:ignore
    st.info('This is an optional step to filter the data')
    st.success(f'Selected variables:')

# Sidebar parameters
with st.sidebar:
    # Slider to set number of plots in a row
    st.session_state.plots_per_row = st.slider(
        "Plots per row",
        1,
        st.session_state.max_plots_per_row,
        st.session_state.plots_per_row,
        key="a_per_page",
        disabled = flag_disabled
    )

    # Checkbox to show/hide plot options
    flag_plot_settings = st.checkbox("Hide plot settings", disabled = flag_disabled)

    # Checkbox to show/hide mri image
    flag_show_img = st.checkbox("Show image", disabled = flag_disabled)
    
    if st.session_state.sel_mrid != '':
        st.sidebar.success("Selected subject: " + st.session_state.sel_mrid)
        
    if st.session_state.sel_roi != '':
        st.sidebar.success("Selected ROI: " + st.session_state.sel_roi)
    

# Panel for plots
flag_disabled = df.shape[0] == 0
with st.expander(":material/monitoring: Plot data", expanded=False):

    # Button to add a new plot
    if st.button("Add plot", disabled = flag_disabled):
        
        # Select xvar and yvar, if not set yet
        num_cols = df.select_dtypes(include='number').columns
        if st.session_state.plot_xvar == '':
            st.session_state.plot_xvar = num_cols[0]
        if st.session_state.plot_yvar == '':
            st.session_state.plot_yvar = num_cols[1]
        
        utilpl.add_plot()
        
    # Read plot ids
    df_p = st.session_state.plots
    list_plots = df_p.index.tolist()
    plots_per_row = st.session_state.plots_per_row

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
                    not flag_plot_settings,
                    st.session_state.sel_mrid
                )
                plots_arr.append(new_plot)

# Show mri image
if flag_show_img:

    # Check if data point selected
    if st.session_state.sel_mrid == "":
        st.warning("Please select a subject on the plot!")

    else:
        if not utilvi.check_image_underlay() or not utilvi.check_image_overlay():
            utilvi.get_image_paths()
                
    if not st.session_state.sel_mrid == "":
        with st.spinner("Wait for it..."):

            # Get selected y var
            sel_var = st.session_state.plots.loc[st.session_state.plot_active, "yvar"]

            # If roi dictionary was used, detect index
            if st.session_state.roi_dict_rev is not None:
                sel_var = st.session_state.roi_dict_rev[sel_var]

            print(sel_var)

            # Check if index exists in overlay mask
            is_in_mask = False
            if os.path.exists(st.session_state.paths["sel_seg"]):
                is_in_mask = utilni.check_roi_index(st.session_state.paths["sel_seg"], sel_var)

            if is_in_mask:
                list_rois = [int(sel_var)]

            else:
                list_rois = utilmuse.get_derived_rois(
                    sel_var,
                    st.session_state.dicts["muse_derived"],
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

            crop_to_mask = False
            is_show_overlay = True
            list_orient = utilni.img_views

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

        # Create a list of checkbox options
        list_orient = st.multiselect("Select viewing planes:", utilni.img_views, utilni.img_views)

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True)



if st.session_state.debug_show_state:
    with st.expander("DEBUG: Session state - all variables"):
        st.write(st.session_state)

if st.session_state.debug_show_paths:
    with st.expander("DEBUG: Session state - paths"):
        st.write(st.session_state.paths)

if st.session_state.debug_show_flags:
    with st.expander("DEBUG: Session state - flags"):
        st.write(st.session_state.flags)    
