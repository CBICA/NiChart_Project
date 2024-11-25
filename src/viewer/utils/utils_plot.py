import os
from typing import Any, Optional

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import utils.utils_trace as utiltr


def add_plot() -> None:
    """
    Adds a new plot (updates a dataframe with plot ids)
    """
    df_p = st.session_state.plots
    plot_id = f"Plot{st.session_state.plot_index}"

    df_p.loc[plot_id] = [
        plot_id,
        st.session_state.plot_var["plot_type"],
        st.session_state.plot_var["xvar"],
        st.session_state.plot_var["yvar"],
        st.session_state.plot_var["hvar"],
        st.session_state.plot_var["hvals"],
        st.session_state.plot_var["corr_icv"],
        st.session_state.plot_var["plot_centiles"],
        st.session_state.plot_var["trend"],
        st.session_state.plot_var["lowess_s"],
        st.session_state.plot_var["traces"],
        st.session_state.plot_var["centtype"],
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


def get_index_in_list(in_list: list, in_item: str) -> Optional[int]:
    """
    Returns the index of the item in list, or None if item not found
    """
    if in_item not in in_list:
        return None
    else:
        return in_list.index(in_item)


def add_plot_tabs(
    df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str
) -> pd.DataFrame:

    ptabs = st.tabs(["Settings", "Layers", ":material/x:"])

    # Tab 1: Plot settings
    with ptabs[0]:
        # Get df columns
        list_cols = df.columns.to_list()
        list_cols_ext = [""] + list_cols
        list_trends = st.session_state.plot_const["trend_types"]

        # Set plotting variables
        xind = get_index_in_list(list_cols, df_plots.loc[plot_id, "xvar"])
        xvar = st.selectbox("X Var", list_cols, key=f"plot_xvar_{plot_id}", index=xind)
        if xvar is not None:
            df_plots.loc[plot_id, "xvar"] = xvar

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            yind = get_index_in_list(list_cols, df_plots.loc[plot_id, "yvar"])
            yvar = st.selectbox(
                "Y Var", list_cols, key=f"plot_yvar_{plot_id}", index=yind
            )
            if yvar is not None:
                df_plots.loc[plot_id, "yvar"] = yvar

        hind = get_index_in_list(list_cols_ext, df_plots.loc[plot_id, "hvar"])
        hvar = st.selectbox(
            "Group by", list_cols_ext, key=f"plot_hvar_{plot_id}", index=hind
        )
        if hvar is not None:
            df_plots.loc[plot_id, "hvar"] = hvar

        if "ICV" in list_cols:
            df_plots.loc[plot_id, "corr_icv"] = st.checkbox(
                "Correct ICV",
                value=df_plots.loc[plot_id, "corr_icv"],
                help="Correct regional volumes using the intra-cranial volume to account for differences in head size",
                key=f"key_check_icv_{plot_id}",
            )

        df_plots.loc[plot_id, "plot_centiles"] = st.checkbox(
            "Plot Centiles",
            value=df_plots.loc[plot_id, "plot_centiles"],
            help="Show centile values for the ROI",
            key=f"key_check_centiles_{plot_id}",
        )

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            tind = get_index_in_list(list_trends, df_plots.loc[plot_id, "trend"])
            trend = st.selectbox(
                "Trend Line",
                list_trends,
                key=f"trend_type_{plot_id}",
                index=tind,
            )
            if trend is not None:
                df_plots.loc[plot_id, "trend"] = trend

            if trend == "Linear":
                df_plots.at[plot_id, "traces"] = ["data", "lin_fit"]
                # df_plots.at[plot_id, 'traces'] = ['data']

            if trend == "Smooth LOWESS Curve":
                df_plots.loc[plot_id, "lowess_s"] = st.slider(
                    "Smoothness", min_value=0.4, max_value=1.0, value=0.7, step=0.1
                )

        if df_plots.loc[plot_id, "plot_type"] == "Distribution Plot":
            df_plots.at[plot_id, "traces"] = ["density", "rug"]

    # Tab 2: Layers
    with ptabs[1]:

        if df_plots.loc[plot_id, "hvar"] != "":
            vals_hue = sorted(df[hvar].unique())
            df_plots.at[plot_id, "hvals"] = st.multiselect(
                "Select groups", vals_hue, vals_hue, key=f"key_select_huevals_{plot_id}"
            )

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            if df_plots.loc[plot_id, "trend"] == "Linear":
                df_plots.at[plot_id, "traces"] = st.multiselect(
                    "Select traces",
                    st.session_state.plot_const["linfit_trace_types"],
                    df_plots.loc[plot_id, "traces"],
                    key=f"key_sel_trace_linfit_{plot_id}",
                )

            # Get plot params
            centtype = df_plots.loc[plot_id, "centtype"]

            # Select plot params from the user
            centind = st.session_state.plot_const["centile_types"].index(centtype)

            centtype = st.selectbox(
                "Centile Type",
                st.session_state.plot_const["centile_types"],
                key=f"cent_type_{plot_id}",
                index=centind,
            )

            # Set plot params to session_state
            st.session_state.plots.loc[plot_id, "centtype"] = centtype

        if df_plots.loc[plot_id, "plot_type"] == "Distribution Plot":
            df_plots.at[plot_id, "traces"] = st.multiselect(
                "Select traces",
                st.session_state.plot_const["distplot_trace_types"],
                df_plots.loc[plot_id, "traces"],
                key=f"key_sel_trace_densityplot_{plot_id}",
            )

    # Tab 3: Reset parameters and/or delete plot
    with ptabs[2]:
        st.button(
            "Delete Plot",
            key=f"p_delete_{plot_id}",
            on_click=remove_plot,
            args=[plot_id],
        )
    return df


def display_scatter_plot(
    df: pd.DataFrame, plot_id: str, show_settings: bool, sel_mrid: str
) -> Any:
    """
    Displays the plot with the plot_id
    """

    def callback_plot_clicked() -> None:
        """
        Set the active plot id to plot that was clicked
        """
        st.session_state.plot_active = plot_id
        # st.rerun()

    # Main container for the plot
    with st.container(border=True):

        # Tabs for plot parameters
        df_filt = df
        if show_settings:
            df_filt = add_plot_tabs(df, st.session_state.plots, plot_id)

        curr_plot = st.session_state.plots.loc[plot_id]

        # Main plot
        layout = go.Layout(
            # height=st.session_state.plot_const['h_init']
            height=st.session_state.plot_const["h_init"]
            * st.session_state.plot_var["h_coeff"],
            margin=dict(
                l=st.session_state.plot_const["margin"],
                r=st.session_state.plot_const["margin"],
                t=st.session_state.plot_const["margin"],
                b=st.session_state.plot_const["margin"],
            ),
        )
        fig = go.Figure(layout=layout)

        # If user selected to use ICV corrected data
        yvar = curr_plot["yvar"]
        if curr_plot["corr_icv"]:
            df_filt[f"{yvar}_corrICV"] = (
                df_filt[yvar] / df_filt["ICV"] * st.session_state.mean_icv
            )
            yvar = f"{yvar}_corrICV"

        # If user selected to plot centiles
        if curr_plot["plot_centiles"]:
            yvar = f'{curr_plot["yvar"]}_centile'

        # Add axis labels
        fig.update_layout(
            xaxis_title=curr_plot["xvar"],
            yaxis_title=yvar,
        )

        # Add data scatter
        utiltr.scatter_trace(
            df_filt,
            curr_plot["xvar"],
            yvar,
            curr_plot["hvar"],
            curr_plot["hvals"],
            curr_plot["traces"],
            st.session_state.plot_var["hide_legend"],
            fig,
        )

        # Add regression lines
        if curr_plot["trend"] == "Linear":
            utiltr.linreg_trace(
                df_filt,
                curr_plot["xvar"],
                yvar,
                curr_plot["hvar"],
                curr_plot["hvals"],
                curr_plot["traces"],
                st.session_state.plot_var["hide_legend"],
                fig,
            )
        elif curr_plot["trend"] == "Smooth LOWESS Curve":
            utiltr.lowess_trace(
                df_filt,
                curr_plot["xvar"],
                yvar,
                curr_plot["hvar"],
                curr_plot["hvals"],
                curr_plot["lowess_s"],
                st.session_state.plot_var["hide_legend"],
                fig,
            )

        # Add centile values
        if curr_plot["centtype"] != "":
            fcent = os.path.join(
                st.session_state.paths["root"],
                "resources",
                "centiles",
                # f"centiles_{curr_plot['centtype']}.csv",
                f"istag_centiles_{curr_plot['centtype']}.csv",
            )
            df_cent = pd.read_csv(fcent)
            utiltr.percentile_trace(df_cent, curr_plot["xvar"], curr_plot["yvar"], fig)

        # Highlight selected data point
        if sel_mrid != "":
            yvar = curr_plot["yvar"]
            if curr_plot["plot_centiles"]:
                yvar = f"{yvar}_centile"
            elif curr_plot["corr_icv"]:
                yvar = f"{yvar}_corrICV"
            utiltr.dot_trace(
                df,
                sel_mrid,
                curr_plot["xvar"],
                yvar,
                st.session_state.plot_var["hide_legend"],
                fig,
            )

        # Catch clicks on plot
        # - on_select: when clicked it will rerun and return the info
        sel_info = st.plotly_chart(
            fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked
        )

        # Detect MRID from the click info and save to session_state
        hind = get_index_in_list(df.columns.tolist(), curr_plot["hvar"])
        if len(sel_info["selection"]["points"]) > 0:
            sind = sel_info["selection"]["point_indices"][0]
            if hind is None:
                sel_mrid = df_filt.iloc[sind]["MRID"]
            else:
                lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                sel_mrid = df_filt[df_filt[curr_plot["hvar"]] == lgroup].iloc[sind][
                    "MRID"
                ]
            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, "yvar"]
            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi
            st.session_state.sel_roi_img = sel_roi
            st.session_state.paths["sel_img"] = ""
            st.session_state.paths["sel_seg"] = ""
            st.rerun()

        return fig


def display_dist_plot(
    df: pd.DataFrame, plot_id: str, show_settings: bool, sel_mrid: str
) -> Any:
    """
    Displays the plot with the plot_id
    """
    # Main container for the plot
    with st.container(border=True):

        # Tabs for plot parameters
        df_filt = df
        if show_settings:
            df_filt = add_plot_tabs(df, st.session_state.plots, plot_id)

        curr_plot = st.session_state.plots.loc[plot_id]

        # Main plot
        fig = utiltr.dist_plot(
            df_filt,
            curr_plot["xvar"],
            curr_plot["hvar"],
            curr_plot["hvals"],
            curr_plot["traces"],
            st.session_state.plot_const["distplot_binnum"],
            st.session_state.plot_var["hide_legend"],
        )

        fig.update_layout(
            # height=st.session_state.plot_const['h_init']
            height=st.session_state.plot_const["h_init"]
            * st.session_state.plot_var["h_coeff"],
            margin=dict(
                l=st.session_state.plot_const["margin"],
                r=st.session_state.plot_const["margin"],
                t=st.session_state.plot_const["margin"],
                b=st.session_state.plot_const["margin"],
            ),
        )
        st.plotly_chart(fig, key=f"key_chart_{plot_id}")

        return fig
