import os

import traceback

import pandas as pd
import plotly.express as px
import streamlit as st
import utils.utils_muse as utilmuse
import utils.utils_nifti as utilni
import utils.utils_st as utilst
import utils.utils_trace as utilstrace
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

VIEWS = ["axial", "coronal", "sagittal"]

# hide_pages(["Image Processing", "Data Analytics"])


def add_plot() -> None:
    """
    Adds a new plot (updates a dataframe with plot ids)
    """
    df_p = st.session_state.plots
    plot_id = f"Plot{st.session_state.plot_index}"
    df_p.loc[plot_id] = [
        plot_id,
        st.session_state.plot_xvar,
        st.session_state.plot_yvar,
        st.session_state.plot_hvar,
        st.session_state.plot_trend,
        st.session_state.plot_centtype,
    ]
    st.session_state.plot_index += 1


# Remove a plot
def remove_plot(plot_id: str) -> None:
    """
    Removes the plot with the plot_id (updates the plot ids dataframe)
    """
    df_p = st.session_state.plots
    df_p = df_p[df_p.pid != plot_id]
    st.session_state.plots = df_p


def display_plot(df: pd.DataFrame, plot_id: str) -> None:
    """
    Displays the plot with the plot_id
    """

    def callback_plot_clicked() -> None:
        """
        Set the active plot id to plot that was clicked
        """
        st.session_state.plot_active = plot_id

    # Create a copy of dataframe for filtered data
    df_filt = df.copy()

    # Main container for the plot
    with st.container(border=True):

        with st.container(border=True):

            # Tabs for parameters
            ptabs = st.tabs(
                [
                    ":lock:",
                    ":large_orange_circle:",
                    ":large_yellow_circle:",
                    ":large_green_circle:",
                    ":x:",
                ]
            )

            # Tab 0: to hide other tabs

            # Tab 1: to set plotting parameters
            with ptabs[1]:
                st.selectbox(
                    "Plot Type", ["DistPlot", "RegPlot"], key=f"plot_type_{plot_id}"
                )

                # Get df columns
                list_cols = df.columns.to_list()

                # Get default plot params
                if st.session_state.plots.loc[plot_id].xvar not in list_cols:
                    if st.session_state.plot_default_xvar in list_cols:
                        st.session_state.plots.loc[plot_id].xvar = st.session_state.plot_default_xvar
                    else:
                        st.session_state.plots.loc[plot_id].xvar = list_cols[1]

                if st.session_state.plots.loc[plot_id].yvar not in list_cols:
                    if st.session_state.plot_default_yvar in list_cols:
                        st.session_state.plots.loc[plot_id].yvar = st.session_state.plot_default_yvar
                    else:
                        st.session_state.plots.loc[plot_id].yvar = list_cols[2]

                if st.session_state.plots.loc[plot_id].hvar not in list_cols:
                    if st.session_state.plot_default_hvar in list_cols:
                        st.session_state.plots.loc[plot_id].hvar = st.session_state.plot_default_hvar
                    else:
                        st.session_state.plots.loc[plot_id].hvar = ''

                xvar = st.session_state.plots.loc[plot_id].xvar
                yvar = st.session_state.plots.loc[plot_id].yvar
                hvar = st.session_state.plots.loc[plot_id].hvar
                trend = st.session_state.plots.loc[plot_id].trend

                # Select plot params from the user
                xind = df.columns.get_loc(xvar)
                yind = df.columns.get_loc(yvar)
                if hvar != '':
                    hind = df.columns.get_loc(hvar)
                else:
                    hind = None
                tind = st.session_state.trend_types.index(trend)

                xvar = st.selectbox(
                    "X Var", df_filt.columns, key=f"plot_xvar_{plot_id}", index=xind
                )
                yvar = st.selectbox(
                    "Y Var", df_filt.columns, key=f"plot_yvar_{plot_id}", index=yind
                )
                hvar = st.selectbox(
                    "Hue Var", df_filt.columns, key=f"plot_hvar_{plot_id}", index=hind
                )
                trend = st.selectbox(
                    "Trend Line",
                    st.session_state.trend_types,
                    key=f"trend_type_{plot_id}",
                    index=tind,
                )

                # Set plot params to session_state
                st.session_state.plots.loc[plot_id].xvar = xvar
                st.session_state.plots.loc[plot_id].yvar = yvar
                st.session_state.plots.loc[plot_id].hvar = hvar
                st.session_state.plots.loc[plot_id].trend = trend

            # Tab 2: to set data filtering parameters
            with ptabs[2]:
                df_filt = filter_dataframe(df, plot_id)

            # Tab 3: to set centiles
            with ptabs[3]:

                # Get plot params
                centtype = st.session_state.plots.loc[plot_id].centtype

                # Select plot params from the user
                centind = st.session_state.cent_types.index(centtype)

                centtype = st.selectbox(
                    "Centile Type",
                    st.session_state.cent_types,
                    key=f"cent_type_{plot_id}",
                    index=centind,
                )

                # Set plot params to session_state
                st.session_state.plots.loc[plot_id].centtype = centtype

            # Tab 4: to reset parameters or to delete plot
            with ptabs[4]:
                st.button(
                    "Delete Plot",
                    key=f"p_delete_{plot_id}",
                    on_click=remove_plot,
                    args=[plot_id],
                )

        # Main plot
        if trend == "none":
            scatter_plot = px.scatter(df_filt, x=xvar, y=yvar, color=hvar)
        else:
            scatter_plot = px.scatter(
                df_filt, x=xvar, y=yvar, color=hvar, trendline=trend
            )

        # Add centile values
        if centtype != "none":
            fcent = os.path.join(
                st.session_state.paths["root"],
                "resources",
                "centiles",
                f"centiles_{centtype}.csv",
            )
            df_cent = pd.read_csv(fcent)
            utilstrace.percentile_trace(df_cent, xvar, yvar, scatter_plot)

        # Add plot
        # - on_select: when clicked it will rerun and return the info
        sel_info = st.plotly_chart(
            scatter_plot, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked
        )

        # Detect MRID from the click info and save to session_state
        if len(sel_info["selection"]["points"]) > 0:

            sind = sel_info["selection"]["point_indices"][0]

            if hind is None:
                sel_mrid = df_filt.iloc[sind]["MRID"]
            else:
                lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                sel_mrid = df_filt[df_filt[hvar] == lgroup].iloc[sind]["MRID"]

            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, "yvar"]

            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi

            st.sidebar.success("Selected subject: " + sel_mrid)
            st.sidebar.success("Selected ROI: " + sel_roi)


def filter_dataframe(df: pd.DataFrame, plot_id: str) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    # Create filters selected by the user
    modification_container = st.container()
    with modification_container:
        widget_no = plot_id + "_filter"
        to_filter_columns = st.multiselect(
            "Filter dataframe on", df.columns, key=widget_no
        )
        for vno, column in enumerate(to_filter_columns):
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                widget_no = plot_id + "_col_" + str(vno)
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=widget_no,
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    # Print sample size after filtering
    dim1, dim2 = df.shape
    st.success("Sample size is: " + str(dim1))

    return df


# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input data
with st.expander(":material/upload: Select or upload input data", expanded=False):

    # Set default path for the plot csv
    if os.path.exists(st.session_state.paths["csv_mlscores"]):
        st.session_state.paths["csv_plots"] = st.session_state.paths["csv_mlscores"]
    elif os.path.exists(st.session_state.paths["csv_seg"]):
        st.session_state.paths["csv_plots"] = st.session_state.paths["csv_seg"]

    if os.path.exists(st.session_state.paths["csv_plots"]):
        st.success(f'Input file detected! Using: {st.session_state.paths["csv_plots"]}')

    # Input csv
    helpmsg = "Input csv file with segmented ROI volumes.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it"
    csv_plots, csv_path = utilst.user_input_file(
        "Select file",
        "btn_input_seg",
        "ROI data file",
        st.session_state.paths["last_in_dir"],
        st.session_state.paths["csv_plots"],
        helpmsg,
    )
    if os.path.exists(csv_plots):
        st.session_state.paths["csv_plots"] = csv_plots
        st.session_state.paths["last_in_dir"] = csv_path

    # Input ROI dict
    helpmsg = "ROI dictionary to rename ROI indices. Should include two columns: ROI_Name, ROI_Index.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it"
    csv_roidict, csv_path = utilst.user_input_file(
        "Select file",
        "btn_input_roidict",
        "ROI dictionary",
        st.session_state.paths["last_in_dir"],
        st.session_state.paths["csv_roidict"],
        helpmsg,
    )
    if os.path.exists(csv_roidict):
        st.session_state.paths["csv_roidict"] = csv_roidict
        st.session_state.paths["last_in_dir"] = csv_path


# Page controls in side bar
with st.sidebar:
    df = pd.DataFrame()
    if os.path.exists(st.session_state.paths["csv_plots"]):
        # Read input csv
        df = pd.read_csv(st.session_state.paths["csv_plots"])
        
        # Apply roi dict to rename columns
        try:
            df_dict = pd.read_csv(st.session_state.paths["csv_roidict"])
            dict_r1 = dict(zip(df_dict['ROI_Index'].astype(str), df_dict['ROI_Name'].astype(str)))
            dict_r2 = dict(zip(df_dict['ROI_Name'].astype(str), df_dict['ROI_Index'].astype(str)))
            st.session_state.roi_dict = dict_r1
            st.session_state.roi_dict_rev = dict_r2
            df = df.rename(columns = dict_r1)
            
        except Exception as e:
            st.warning('Could not rename columns using roi dict!')
        
        with st.container(border=True):
            # Slider to set number of plots in a row
            st.session_state.plots_per_row = st.slider(
                "Plots per row",
                1,
                st.session_state.max_plots_per_row,
                st.session_state.plots_per_row,
                key="a_per_page",
            )

        with st.container(border=True):

            st.write("Plot Settings")

            # Tabs for parameters
            ptabs = st.tabs(
                [
                    ":lock:",
                    ":large_orange_circle:",
                    ":large_yellow_circle:",
                    ":large_green_circle:",
                ]
            )

            # Tab 0: to set plotting parameters
            with ptabs[1]:
                # Default values for plot params
                st.session_state.plot_hvar = "Sex"

                plot_xvar_ind = 0
                if st.session_state.plot_xvar in df.columns:
                    plot_xvar_ind = df.columns.get_loc(st.session_state.plot_xvar)

                plot_yvar_ind = 0
                if st.session_state.plot_yvar in df.columns:
                    plot_yvar_ind = df.columns.get_loc(st.session_state.plot_yvar)

                plot_hvar_ind = 0
                if st.session_state.plot_hvar in df.columns:
                    plot_hvar_ind = df.columns.get_loc(st.session_state.plot_hvar)

                st.session_state.plot_xvar = st.selectbox(
                    "Default X Var",
                    df.columns,
                    key="plot_xvar_init",
                    index=plot_xvar_ind,
                )
                st.session_state.plot_yvar = st.selectbox(
                    "Default Y Var",
                    df.columns,
                    key="plot_yvar_init",
                    index=plot_yvar_ind,
                )
                st.session_state.sel_var = st.session_state.plot_yvar

                st.session_state.plot_hvar = st.selectbox(
                    "Default Hue Var",
                    df.columns,
                    key="plot_hvar_init",
                    index=plot_hvar_ind,
                )
                trend_index = st.session_state.trend_types.index(
                    st.session_state.plot_trend
                )
                st.session_state.plot_trend = st.selectbox(
                    "Default Trend Line",
                    st.session_state.trend_types,
                    key="trend_type_init",
                    index=trend_index,
                )

# Panel for plots
with st.expander(":material/monitoring: Plot data", expanded=False):

    # Button to add a new plot
    if st.button("Add plot"):
        add_plot()

    # Add a single plot (default: initial page displays a single plot)
    if st.session_state.plots.shape[0] == 0:
        add_plot()

    # Read plot ids
    df_p = st.session_state.plots
    list_plots = df_p.index.tolist()
    plots_per_row = st.session_state.plots_per_row

    # Render plots
    #  - iterates over plots;
    #  - for every "plots_per_row" plots, creates a new columns block, resets column index, and displays the plot
    if df.shape[0] > 0:
        for i, plot_ind in enumerate(list_plots):
            column_no = i % plots_per_row
            if column_no == 0:
                blocks = st.columns(plots_per_row)
            with blocks[column_no]:
                display_plot(df, plot_ind)

placeholder_imgview = st.empty()

with st.expander(":material/settings: Viewer settings", expanded=False):

    # Create a list of checkbox options
    list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS)

    # View hide overlay
    is_show_overlay = st.checkbox("Show overlay", True)

    # View hide overlay
    crop_to_mask = st.checkbox("Crop to mask", True)

# Panel for selecting input folders for images
with st.expander(":material/upload: Viewer input folders"):
    # Input T1 image folder
    helpmsg = "Folder with T1 images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_t1 = utilst.user_input_folder(
        "Select folder",
        "btn_indir_t1",
        "Input T1 image folder",
        st.session_state.paths["last_in_dir"],
        st.session_state.paths["T1"],
        helpmsg,
    )
    st.session_state.paths["T1"] = path_t1

    # Input DLMUSE image folder
    helpmsg = "Folder with DLMUSE images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_seg = utilst.user_input_folder(
        "Select folder",
        "btn_indir_seg",
        "Input label image folder",
        st.session_state.paths["last_in_dir"],
        st.session_state.paths["DLMUSE"],
        helpmsg,
    )
    st.session_state.paths["DLMUSE"] = path_seg

    # T1 suffix
    suff_t1img = utilst.user_input_text(
        "T1 image suffix", st.session_state.suff_t1img, helpmsg
    )
    st.session_state.suff_t1img = suff_t1img

    # DLMUSE suffix
    suff_seg = utilst.user_input_text(
        "Label image suffix", st.session_state.suff_seg, helpmsg
    )
    st.session_state.suff_seg = suff_seg

# Panel for viewing images and segmentations
with placeholder_imgview.expander(":material/visibility: View segmentations", expanded=False):

    flag_show = True
    if st.session_state.sel_mrid == "":
        st.warning("Please select a subject on the plot!")
        flag_show = False
    else:
        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths["T1"],
            st.session_state.sel_mrid + st.session_state.suff_t1img,
        )
        st.session_state.paths["sel_seg"] = os.path.join(
            st.session_state.paths["DLMUSE"],
            st.session_state.sel_mrid + st.session_state.suff_seg,
        )
        if not os.path.exists(st.session_state.paths["sel_img"]):
            st.warning(
                f"Could not find underlay image: {st.session_state.paths['sel_img']}"
            )
            #st.error('Underlay image not found. Please check input path and suffix in the panel below!')
            flag_show = False

        if not os.path.exists(st.session_state.paths["sel_seg"]):
            st.warning(
                f"Could not find overlay image: {st.session_state.paths['sel_seg']}"
            )
            flag_show = False

    if flag_show:
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
                        ind_view = VIEWS.index(tmp_orient)
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


with st.expander("FIXME: TMP - Session state"):
    st.write(st.session_state)
with st.expander("TMP: session vars - paths"):
    st.write(st.session_state.paths)
