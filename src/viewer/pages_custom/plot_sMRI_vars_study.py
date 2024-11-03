import os
import glob
import pandas as pd
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_trace as utiltr
import utils.utils_st as utilst
import utils.utils_viewimg as utilvi
import utils.utils_plot as utilpl
from stqdm import stqdm

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Set default path for the data csv
if not os.path.exists(st.session_state.paths["csv_plot"]):
    if os.path.exists(st.session_state.paths["csv_mlscores"]):
        st.session_state.paths["csv_plot"] = st.session_state.paths["csv_mlscores"]
    elif os.path.exists(st.session_state.paths["csv_dlmuse"]):
        st.session_state.paths["csv_plot"] = st.session_state.paths["csv_dlmuse"]

# Panel for selecting input csv
flag_disabled = not st.session_state.flags['dset']

if st.session_state.app_type == "CLOUD":
    with st.expander(":material/upload: Upload data", expanded=False):  # type:ignore
        utilst.util_upload_file(
            st.session_state.paths["csv_plot"],
            "Input data csv file",
            "key_in_csv",
            flag_disabled,
            "visible",
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")

else:  # st.session_state.app_type == 'DESKTOP'
    with st.expander(":material/upload: Select data", expanded=False):
        utilst.util_select_file(
            "selected_data_file",
            "Data csv",
            st.session_state.paths["csv_plot"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_plot"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_plot"]})", icon=":material/thumb_up:")

# Input csv
flag_disabled = not os.path.exists(st.session_state.paths['csv_plot'])

df = st.session_state.df_plot
if os.path.exists(st.session_state.paths["csv_plot"]):
    # Read input csv
    if st.session_state.df_plot.shape[0] == 0:
        st.session_state.df_plot = pd.read_csv(st.session_state.paths["csv_plot"])

    # Apply roi dict to rename columns
    try:
        df_dict = pd.read_csv(st.session_state.paths["csv_roidict"])
        dict_r1 = dict(
            zip(df_dict["ROI_Index"].astype(str), df_dict["ROI_Name"].astype(str))
        )
        dict_r2 = dict(
            zip(df_dict["ROI_Name"].astype(str), df_dict["ROI_Index"].astype(str))
        )
        st.session_state.roi_dict = dict_r1
        st.session_state.roi_dict_rev = dict_r2
        df = df.rename(columns=dict_r1)

    except Exception:
        print("Could not rename columns using roi dict!")

    df = st.session_state.df_plot

# Panel for selecting variables
flag_disabled = not st.session_state.flags['dset']
with st.expander(":material/upload: Select Vars", expanded=False):  # type:ignore

    var_cats = { 'CatA': ['MRID', 'Age', 'Sex'], 'CatB': ['GM', 'WM', ], 'CatC': ['Ventricles', 'SPAREAD'] }

    # User selects a category to include
    sel_cat = st.selectbox('Select category', list(var_cats.keys()), index = None)

    if sel_cat is None:
        sel_vars = []
    else:
        sel_vars = var_cats[sel_cat]

    sel_vars = st.multiselect('Which ones to keep?', sel_vars, sel_vars)
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
        st.session_state.df_plot = df[st.session_state.plot_sel_vars]

        print(df)

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

    if st.session_state.plot_xvar == '':
        xvar = st.selectbox(
            "X Var", df.columns, key="key_plot_xvar", index=None, disabled = flag_disabled
        )
        if xvar is not None:
            st.session_state.plot_xvar = xvar
    if st.session_state.plot_yvar == '':
        yvar = st.selectbox(
            "Y Var", df.columns, key="key_plot_yvar", index=None, disabled = flag_disabled
        )
        if yvar is not None:
            st.session_state.plot_yvar = yvar
    if st.session_state.plot_hvar == '':
        hvar = st.selectbox(
            "Hue Var", df.columns, key="key_plot_hvar", index=None, disabled = flag_disabled
        )
        if hvar is not None:
            st.session_state.plot_hvar = hvar

    # Checkbox to show/hide plot options
    flag_plot_settings = st.checkbox("Show plot settings", disabled = flag_disabled)

    # Checkbox to show/hide mri image
    flag_show_img = st.checkbox("Show image", disabled = flag_disabled)

    # Button to add a new plot
    if st.button("Add plot", disabled = flag_disabled):
        utilpl.add_plot()


# Panel for plots
with st.expander(":material/monitoring: Plot data", expanded=False):

    if not flag_disabled:

        # Read plot ids
        df_p = st.session_state.plots
        list_plots = df_p.index.tolist()
        plots_per_row = st.session_state.plots_per_row

        # Render plots
        #  - iterates over plots;
        #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
        if df.shape[0] > 0:
            plots_arr = []
            for i, plot_ind in stqdm(
                enumerate(list_plots), desc="Rendering plots ...", total=len(list_plots)
            ):
                column_no = i % plots_per_row
                if column_no == 0:
                    blocks = st.columns(plots_per_row)
                with blocks[column_no]:
                    new_plot = utilpl.display_plot(
                        df,
                        plot_ind,
                        flag_plot_settings,
                        st.session_state.sel_mrid
                    )
                    plots_arr.append(new_plot)

            # # Add marker for selected mrid to each plot
            # if st.session_state.sel_mrid != '':
            #     for i, tmp_plot in enumerate(plots_arr):
            #         utiltr.selid_trace(
            #             df,
            #             st.session_state.sel_mrid,
            #             df_p.iloc[i].xvar,
            #             df_p.iloc[i].yvar,
            #             tmp_plot
            #         )

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
