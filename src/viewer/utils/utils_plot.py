from typing import Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

def add_plot_tabs(df: pd.DataFrame, plot_id: str) -> pd.DataFrame:
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
            
        return df_filt, trend, xvar, yvar, hind, hvar, centtype


def display_plot(df: pd.DataFrame, plot_id: str) -> None:
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
        df_filt, trend, xvar, yvar, hind, hvar, centtype = add_plot_tabs(df, plot_id)
        
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
