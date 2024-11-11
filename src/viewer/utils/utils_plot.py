import os

import pandas as pd
import plotly.express as px
import streamlit as st
import utils.utils_dataframe as utilsdf
import utils.utils_trace as utiltr
import plotly.graph_objs as go

def add_plot() -> None:
    """
    Adds a new plot (updates a dataframe with plot ids)
    """
    df_p = st.session_state.plots
    plot_id = f"Plot{st.session_state.plot_index}"

    df_p.loc[plot_id] = [
        plot_id,
        st.session_state.plot_var['xvar'],
        st.session_state.plot_var['yvar'],
        st.session_state.plot_var['hvar'],
        st.session_state.plot_var['hvals'],
        st.session_state.plot_var['trend'],
        st.session_state.plot_var['lowess_s'],
        st.session_state.plot_var['traces'],
        st.session_state.plot_var['centtype'],
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

def get_index_in_list(in_list, in_item):
    '''
    Returns the index of the item in list, or None if item not found
    '''
    if in_item not in in_list:
        return None
    else:
        return in_list.index(in_item)

def add_plot_tabs(df: pd.DataFrame, plot_id: str) -> pd.DataFrame:

    ptabs = st.tabs(["Plot Type", "Settings", "Layers", ":x:"])

    # Tab 1: Plot type
    with ptabs[0]:
        st.selectbox(
            "Plot Type", ["Scatter Plot"], key=f"plot_type_{plot_id}"
        )

    # Tab 2: Plot settings
    with ptabs[1]:
        # Get df columns
        list_cols = df.columns.to_list()
        list_cols_ext = [''] + list_cols
        list_trends = st.session_state.plot_const['trend_types']

        # Set plotting variables
        xind = get_index_in_list(list_cols, st.session_state.plots.loc[plot_id].xvar)
        xvar = st.selectbox(
            "X Var", list_cols, key=f"plot_xvar_{plot_id}", index=xind
        )
        yind = get_index_in_list(list_cols, st.session_state.plots.loc[plot_id].yvar)
        yvar = st.selectbox(
            "Y Var", list_cols, key=f"plot_yvar_{plot_id}", index=yind
        )
        hind = get_index_in_list(list_cols_ext, st.session_state.plots.loc[plot_id].hvar)
        hvar = st.selectbox(
            "Group by", list_cols_ext, key=f"plot_hvar_{plot_id}", index=hind
        )

        tind = get_index_in_list(list_trends, st.session_state.plots.loc[plot_id].trend)
        trend = st.selectbox(
            "Trend Line", list_trends, key=f"trend_type_{plot_id}", index=tind,
        )
        if trend == 'Smooth LOWESS Curve':
            st.session_state.plots.loc[plot_id, 'lowess_s'] = st.slider(
                'Smoothness', min_value=0.4, max_value=1.0, value=0.7, step=0.1
            )

        # Set plot params to session_state
        if xvar is not None:
            st.session_state.plots.loc[plot_id, 'xvar'] = xvar
        if yvar is not None:
            st.session_state.plots.loc[plot_id, 'yvar'] = yvar
        if hvar != '':
            st.session_state.plots.loc[plot_id, 'hvar'] = hvar
        if trend != '':
            st.session_state.plots.loc[plot_id, 'trend'] = trend

        st.session_state.plots.at[plot_id, 'traces'] = ['Data', 'lin']

    # Tab 3: Layers
    with ptabs[2]:

        if hvar != '':
            vals_hue = df[hvar].unique().tolist()
            st.session_state.plots.at[plot_id, 'hvals'] = st.multiselect(
                'Select groups', vals_hue, vals_hue
            )

        if trend == 'Linear':
            st.session_state.plots.at[plot_id, 'traces'] = st.multiselect(
                'Select traces', ['Data', 'lin', 'lin_conf95'], ['Data', 'lin', 'lin_conf95']
            )

        # Get plot params
        centtype = st.session_state.plots.loc[plot_id].centtype

        # Select plot params from the user
        centind = st.session_state.plot_var['centtype'].index(centtype)

        centtype = st.selectbox(
            "Centile Type",
            st.session_state.plot_var['centtype'],
            key=f"cent_type_{plot_id}",
            index=centind,
        )

        # Set plot params to session_state
        st.session_state.plots.loc[plot_id, 'centtype'] = centtype

    # Tab 4: Reset parameters and/or delete plot
    with ptabs[3]:
        st.button(
            "Delete Plot",
            key=f"p_delete_{plot_id}",
            on_click=remove_plot,
            args=[plot_id],
        )
    return df


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
        #st.rerun()

    # Main container for the plot
    with st.container(border=True):

        # Tabs for plot parameters
        df_filt = df
        if show_settings:
            df_filt = add_plot_tabs(df, plot_id)

        [xvar, yvar, hvar, hvals, trend, lowess_s, traces, centtype] = st.session_state.plots.loc[plot_id][['xvar', 'yvar', 'hvar', 'hvals', 'trend', 'lowess_s', 'traces', 'centtype']]
        hind = get_index_in_list(df.columns.tolist(), hvar)

        # Main plot
        layout = go.Layout(
            # height=st.session_state.plot_const['height_init']
            height=st.session_state.plot_const['height_init'] * st.session_state.plot_height_coeff,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig = go.Figure(layout = layout)
        
        # Add axis labels
        fig.update_layout(
            xaxis_title = xvar,
            yaxis_title = yvar,
        )
        
        utiltr.scatter_trace(df_filt, xvar, yvar, hvar, hvals, traces, fig)
        if trend == 'Linear':
            utiltr.linreg_trace(df_filt, xvar, yvar, hvar, hvals, traces, fig)
        elif trend == 'Smooth LOWESS Curve':
            utiltr.lowess_trace(df_filt, xvar, yvar, hvar, hvals, lowess_s, fig)

        # Add centile values
        if centtype != '':
            fcent = os.path.join(
                st.session_state.paths["root"],
                "resources",
                "centiles",
                f"centiles_{centtype}.csv",
            )
            df_cent = pd.read_csv(fcent)
            utiltr.percentile_trace(df_cent, xvar, yvar, fig)

        # Highlight selected data point
        if sel_mrid != '':
            utiltr.selid_trace(df, sel_mrid, xvar, yvar, fig)


        # Catch clicks on plot
        # - on_select: when clicked it will rerun and return the info
        sel_info = st.plotly_chart(
            fig, key=f"bubble_chart_{plot_id}", on_select=callback_plot_clicked
        )

        # Detect MRID from the click info and save to session_state
        if len(sel_info["selection"]["points"]) > 0:

            sind = sel_info["selection"]["point_indices"][0]

            if hind is None:
                sel_mrid = df_filt.iloc[sind]["MRID"]
            else:
                
                print(f'AAA {sel_info}')
                
                lgroup = sel_info["selection"]["points"][0]["legendgroup"]
                sel_mrid = df_filt[df_filt[hvar] == lgroup].iloc[sind]["MRID"]

            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, "yvar"]

            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi

            st.rerun()

        return fig
