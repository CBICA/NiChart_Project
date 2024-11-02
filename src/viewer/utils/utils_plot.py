import os

import pandas as pd
import plotly.express as px
import streamlit as st
import utils.utils_dataframe as utilsdf
import utils.utils_trace as utilstrace
import plotly.graph_objs as go


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


def add_plot_tabs(df: pd.DataFrame, plot_id: str, show_settings: bool) -> pd.DataFrame:

    df_filt = df
    trend = None
    xvar = 'Age'
    yvar = 'GM'
    hind = 0
    hvar = 'Sex'
    centtype = 'none'

    if show_settings:

    # with st.container(border=True):
        # Tabs for parameters
        ptabs = st.tabs(
            [
                ":large_orange_circle:",
                ":large_yellow_circle:",
                ":large_green_circle:",
                ":x:",
            ]
        )

        # Tab 0: to hide other tabs

        # Tab 1: to set plotting parameters
        with ptabs[0]:
            st.selectbox(
                "Plot Type", ["DistPlot", "RegPlot"], key=f"plot_type_{plot_id}"
            )

            # Get df columns
            list_cols = df.columns.to_list()

            # Get default plot params
            if st.session_state.plots.loc[plot_id].xvar not in list_cols:
                if st.session_state.plot_default_xvar in list_cols:
                    st.session_state.plots.loc[plot_id].xvar = (
                        st.session_state.plot_default_xvar
                    )
                else:
                    st.session_state.plots.loc[plot_id].xvar = list_cols[1]

            if st.session_state.plots.loc[plot_id].yvar not in list_cols:
                if st.session_state.plot_default_yvar in list_cols:
                    st.session_state.plots.loc[plot_id].yvar = (
                        st.session_state.plot_default_yvar
                    )
                else:
                    st.session_state.plots.loc[plot_id].yvar = list_cols[2]

            if st.session_state.plots.loc[plot_id].hvar not in list_cols:
                if st.session_state.plot_default_hvar in list_cols:
                    st.session_state.plots.loc[plot_id].hvar = (
                        st.session_state.plot_default_hvar
                    )
                else:
                    st.session_state.plots.loc[plot_id].hvar = ""

            xvar = st.session_state.plots.loc[plot_id].xvar
            yvar = st.session_state.plots.loc[plot_id].yvar
            hvar = st.session_state.plots.loc[plot_id].hvar
            trend = st.session_state.plots.loc[plot_id].trend

            # Select plot params from the user
            xind = df.columns.get_loc(xvar)
            yind = df.columns.get_loc(yvar)
            if hvar != "":
                hind = df.columns.get_loc(hvar)
            else:
                hind = None
            tind = st.session_state.trend_types.index(trend)

            xvar = st.selectbox(
                "X Var", df.columns, key=f"plot_xvar_{plot_id}", index=xind
            )
            yvar = st.selectbox(
                "Y Var", df.columns, key=f"plot_yvar_{plot_id}", index=yind
            )
            hvar = st.selectbox(
                "Hue Var", df.columns, key=f"plot_hvar_{plot_id}", index=hind
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
        with ptabs[1]:
            df_filt = utilsdf.filter_dataframe(df, plot_id)

        # Tab 3: to set centiles
        with ptabs[2]:

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
        with ptabs[3]:
            st.button(
                "Delete Plot",
                key=f"p_delete_{plot_id}",
                on_click=remove_plot,
                args=[plot_id],
            )

    return df_filt, trend, xvar, yvar, hind, hvar, centtype


def display_plot(
    df: pd.DataFrame,
    plot_id: str,
    show_settings: bool,
    sel_mrid: str
) -> None:
    """
    Displays the plot with the plot_id
    """

    def callback_plot_clicked() -> None:
        """
        Set the active plot id to plot that was clicked
        """
        st.session_state.plot_active = plot_id

    # Main container for the plot
    with st.container(border=True):

        # Tabs for plot parameters
        df_filt, trend, xvar, yvar, hind, hvar, centtype = add_plot_tabs(df, plot_id, show_settings)

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

        # Highlight selected data point
        if sel_mrid != '':
            utilstrace.selid_trace(df, sel_mrid, xvar, yvar, scatter_plot)


        # Catch clicks on plot
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
