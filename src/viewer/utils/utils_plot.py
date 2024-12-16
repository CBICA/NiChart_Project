import os
from typing import Any, Optional

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import utils.utils_trace as utiltr

def add_items_to_list(my_list, items_to_add):
  """Adds multiple items to a list, avoiding duplicates.

  Args:
    my_list: The list to add items to.
    items_to_add: A list of items to add.

  Returns:
    The modified list.
  """
  for item in items_to_add:
    if item not in my_list:
      my_list.append(item)
  return my_list

def remove_items_from_list(my_list, items_to_remove):
  """Removes multiple items from a list.

  Args:
    my_list: The list to remove items from.
    items_to_remove: A list of items to remove.

  Returns:
    The modified list.
  """
  out_list=[]
  for item in my_list:
    if item not in items_to_remove:
      out_list.append(item)
  return out_list

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
        st.session_state.plot_var["xmin"],
        st.session_state.plot_var["xmax"],
        st.session_state.plot_var["yvar"],
        st.session_state.plot_var["ymin"],
        st.session_state.plot_var["ymax"],
        st.session_state.plot_var["hvar"],
        st.session_state.plot_var["hvals"],
        st.session_state.plot_var["corr_icv"],
        st.session_state.plot_var["plot_cent_normalized"],
        st.session_state.plot_var["trend"],
        st.session_state.plot_var["lowess_s"],
        st.session_state.plot_var["traces"].copy(),
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

def set_x_bounds(
    df: pd.DataFrame,
    df_plots: pd.DataFrame,
    plot_id: str,
    xvar: str
) -> list:
    # Set x and y min/max if not set
    # Values include some margin added for viewing purposes    
    xmin=df[xvar].min()
    xmax=df[xvar].max()
    dx=xmax-xmin
    if dx==0:       # Margin defined based on the value if delta is 0
        xmin = xmin - xmin/8
        xmax = xmax + xmax/8
    else:           # Margin defined based on the delta otherwise
        xmin = xmin - dx/5
        xmax = xmax + dx/5

    df_plots.loc[plot_id, "xmax"]=xmax
    df_plots.loc[plot_id, "xmin"]=xmin
    
def set_y_bounds(
    df: pd.DataFrame,
    df_plots: pd.DataFrame,
    plot_id: str,
    yvar: str
) -> list:
    # Set x and y min/max if not set
    # Values include some margin added for viewing purposes    
    ymin=df[yvar].min()
    ymax=df[yvar].max()
    dy=ymax-ymin
    if dy==0:       # Margin defined based on the value if delta is 0
        ymin = ymin - ymin/8
        ymax = ymax + ymax/8
    else:           # Margin defined based on the delta otherwise
        ymin = ymin - dy/5
        ymax = ymax + dy/5

    df_plots.loc[plot_id, "ymax"]=ymax
    df_plots.loc[plot_id, "ymin"]=ymin

def add_plot_tabs(
    df: pd.DataFrame, df_plots: pd.DataFrame, plot_id: str
) -> pd.DataFrame:

    # Set xy bounds initially to make plots consistent
    if df_plots.loc[plot_id, "xmin"]==-1:
        set_x_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "xvar"])
    if df_plots.loc[plot_id, "ymin"]==-1:
        set_y_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "yvar"])
    
    ptabs = st.tabs(["Settings", "Layers", ":material/x:"])
    
    # Tab 1: Plot settings
    with ptabs[0]:
        # Get df columns
        list_cols = df.columns[df.columns.str.contains('_centiles')==False].to_list()
        list_cols_ext = [""] + list_cols
        list_trends = st.session_state.plot_const["trend_types"]

      
        # Set x var
        def on_xvar_change():
            key=f"plot_xvar_{plot_id}"
            sel_val=st.session_state[key]
            df_plots.loc[plot_id, "xvar"]=sel_val
            
            # Update x bounds
            set_x_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "xvar"])

            # Reset centiles if x is not Age
            if sel_val != 'Age':
                df_plots.loc[plot_id, "centtype"]=''
                df_plots.at[plot_id, "traces"] = remove_items_from_list(
                    df_plots.loc[plot_id, "traces"],
                    st.session_state.plot_const['centile_trace_types']
                )

        xind = get_index_in_list(list_cols, df_plots.loc[plot_id, "xvar"])        
        st.selectbox(
            "X Var",
            list_cols,
            key=f"plot_xvar_{plot_id}",
            index=xind,
            on_change=on_xvar_change
        )
        
        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            # Set y var
            def on_yvar_change():
                key=f"plot_yvar_{plot_id}"
                sel_val=st.session_state[key]
                df_plots.loc[plot_id, "yvar"]=sel_val

                # Check if centile value exists
                if df_plots.loc[plot_id, "plot_cent_normalized"]:
                    y_new=df_plots.loc[plot_id, "yvar"] + '_centiles'
                    if y_new not in df.columns:
                        df_plots.loc[plot_id, "plot_cent_normalized"] = False

                # Update y bounds
                if df_plots.loc[plot_id, "plot_cent_normalized"]:
                    set_y_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "yvar"]+'_centiles')
                else:
                    set_y_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "yvar"])
                
            yind = get_index_in_list(list_cols, df_plots.loc[plot_id, "yvar"])
            st.selectbox(
                "Y Var",
                list_cols,
                key=f"plot_yvar_{plot_id}",
                index=yind,
                on_change=on_yvar_change
            )

        # Set hvar
        def on_hvar_change():
            key=f"plot_hvar_{plot_id}"
            sel_val=st.session_state[key]
            df_plots.loc[plot_id, "hvar"]=sel_val
            df_plots.at[plot_id, "hvals"]=[]
            
        hind = get_index_in_list(list_cols_ext, df_plots.loc[plot_id, "hvar"])
        st.selectbox(
            "Group by",
            list_cols_ext,
            key=f"plot_hvar_{plot_id}",
            index=hind,
            on_change=on_hvar_change
        )

        if "ICV" in list_cols:
            # Set icv corr flag
            def on_check_icvcorr_change():
                key=f"key_check_icv_{plot_id}"
                sel_val=st.session_state[key]
                df_plots.loc[plot_id, "corr_icv"]=sel_val
                
            st.checkbox(
                "Correct ICV",
                value=df_plots.loc[plot_id, "corr_icv"],
                help="Correct regional volumes using the intra-cranial volume to account for differences in head size",
                key=f"key_check_icv_{plot_id}",
                on_change=on_check_icvcorr_change
            )

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            # Set view cent norm data flag
            def on_check_centnorm_change():
                key=f"key_check_centiles_{plot_id}"
                sel_val=st.session_state[key]
                df_plots.loc[plot_id, "plot_cent_normalized"]=sel_val
                if sel_val:
                    set_y_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "yvar"] + '_centiles')
                else:
                    set_y_bounds(df, df_plots, plot_id, df_plots.loc[plot_id, "yvar"])

            y_new=df_plots.loc[plot_id, "yvar"] + '_centiles'
            if y_new in df.columns:
                st.checkbox(
                    "Plot ROIs normalized by centiles",
                    value=df_plots.loc[plot_id, "plot_cent_normalized"],
                    help="Show ROI values normalized by reference centiles",
                    key=f"key_check_centiles_{plot_id}",
                    on_change=on_check_centnorm_change
                )

            # Select trend
            def on_trend_sel_change():
                key=f"trend_type_{plot_id}"
                sel_val=st.session_state[key]
                df_plots.loc[plot_id, "trend"]=sel_val
                if sel_val == "Linear":
                    df_plots.at[plot_id, "traces"] = add_items_to_list(
                        df_plots.loc[plot_id, "traces"], ["lin_fit"]
                    )
                else:
                    df_plots.at[plot_id, "traces"] = remove_items_from_list(
                        df_plots.loc[plot_id, "traces"], ["lin_fit", "conf_95%"]
                    )
                    
            tind = get_index_in_list(list_trends, df_plots.loc[plot_id, "trend"])            
            st.selectbox(
                "Trend Line",
                list_trends,
                key=f"trend_type_{plot_id}",
                index=tind,
                on_change=on_trend_sel_change
            )

            # Select lowess smoothness            
            if df_plots.loc[plot_id, "trend"]=="Smooth LOWESS Curve":
                def on_sel_lowessval_change():
                    key=f"lowess_sm_{plot_id}"
                    sel_val=st.session_state[key]
                    df_plots.loc[plot_id, "lowess_s"]=sel_val
                
                st.slider(
                    "Smoothness",
                    min_value=0.4,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key=f"lowess_sm_{plot_id}",
                    on_change=on_sel_lowessval_change
                )

        if df_plots.loc[plot_id, "plot_type"] == "Distribution Plot":
            df_plots.at[plot_id, "traces"] = ["density", "rug"]

    # Tab 2: Layers
    with ptabs[1]:

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":
            
            if df_plots.loc[plot_id, "xvar"] != 'Age':
                st.warning('To view data centiles please select the "Age" as the x axis variable!')
            else:
                # Select centile type
                def on_centile_sel_change():
                    # Set centile selection to plot var
                    key=f"cent_type_{plot_id}"
                    sel_val=st.session_state[key]
                    df_plots.loc[plot_id, "centtype"]=sel_val
                    if sel_val == "":
                        df_plots.at[plot_id, "traces"] = remove_items_from_list(
                            df_plots.loc[plot_id, "traces"],
                            st.session_state.plot_const['centile_trace_types']
                        )
                    else:
                        df_plots.at[plot_id, "traces"] = add_items_to_list(
                            df_plots.loc[plot_id, "traces"],
                            st.session_state.plot_const['centile_trace_types']
                        )

                centtype = df_plots.loc[plot_id, "centtype"]
                centind = st.session_state.plot_const["centile_types"].index(centtype)
                sel_options = st.selectbox(
                    "Centile Type",
                    st.session_state.plot_const["centile_types"],
                    key=f"cent_type_{plot_id}",
                    index=centind,
                    on_change=on_centile_sel_change
                )

        # Select group (hue) values
        hvar=df_plots.loc[plot_id, "hvar"]
        if hvar != "":
            def on_hvals_change():
                key=f"key_select_hvals_{plot_id}"
                sel_val=st.session_state[key]
                df_plots.at[plot_id, "hvals"]=sel_val
                #if sel_val != '':
            
            vals_hue = sorted(df[hvar].unique())
            df_plots.at[plot_id, "hvals"] = st.multiselect(
                "Select groups",
                vals_hue,
                vals_hue,
                key=f"key_select_hvals_{plot_id}",
                on_change=on_hvals_change
            )

        if df_plots.loc[plot_id, "plot_type"] == "Scatter Plot":

            ## Update current list of traces
            #df_plots.at[plot_id, "traces"] = [x for x in df_plots.loc[plot_id, "traces"] if x in list_traces]

            # Read user selection for traces
            def on_scatter_trace_sel_change():
                key=f"key_sel_traces_scatter_{plot_id}"
                sel_trace=st.session_state[key]
                df_plots.at[plot_id, "traces"]=sel_trace

            list_traces = ['data']
            if df_plots.loc[plot_id, "trend"] == "Linear":
                list_traces = list_traces + st.session_state.plot_const["linfit_trace_types"]
            if df_plots.loc[plot_id, "centtype"] != "":
                list_traces = list_traces + st.session_state.plot_const["centile_trace_types"]

            sel_trace = st.multiselect(
                "Select traces",
                list_traces,
                df_plots.loc[plot_id, "traces"],
                key=f"key_sel_traces_scatter_{plot_id}",
                on_change=on_scatter_trace_sel_change
            )

        if df_plots.loc[plot_id, "plot_type"] == "Distribution Plot":
            # Read user selection for traces
            def on_dist_trace_sel_change():
                key=f"key_sel_trace_dist_{plot_id}"
                sel_trace=st.session_state[key]
                df_plots.at[plot_id, "traces"]=sel_trace

            df_plots.at[plot_id, "traces"] = st.multiselect(
                "Select traces",
                st.session_state.plot_const["distplot_trace_types"],
                df_plots.loc[plot_id, "traces"],
                key=f"key_sel_trace_dist_{plot_id}",
                on_change=on_dist_trace_sel_change
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
        if curr_plot["plot_cent_normalized"]:
            yvar = f'{curr_plot["yvar"]}_centiles'
            if yvar not in df.columns:
                st.warning(f'Centile values not available for variable: {curr_plot["yvar"]}')
                yvar=curr_plot["yvar"]

        # Add axis labels
        fig.update_layout(
            xaxis_title=curr_plot["xvar"],
            yaxis_title=yvar,
        )

        # Add data scatter
        utiltr.scatter_trace(
            df_filt,
            curr_plot["xvar"],
            curr_plot["xmin"],
            curr_plot["xmax"],
            yvar,
            curr_plot["ymin"],
            curr_plot["ymax"],
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
                curr_plot["xmin"],
                curr_plot["xmax"],
                yvar,
                curr_plot["ymin"],
                curr_plot["ymax"],
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
                curr_plot["xmin"],
                curr_plot["xmax"],
                yvar,
                curr_plot["ymin"],
                curr_plot["ymax"],
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
            utiltr.percentile_trace(
                df_cent,
                curr_plot["xvar"],
                curr_plot["xmin"],
                curr_plot["xmax"],
                yvar,
                curr_plot["ymin"],
                curr_plot["ymax"],
                curr_plot['traces'],
                st.session_state.plot_var["hide_legend"],
                fig
            )

        # Highlight selected data point
        if sel_mrid != "":
            yvar = curr_plot["yvar"]
            if curr_plot["plot_cent_normalized"]:
                yvar = f"{yvar}_centiles"
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
