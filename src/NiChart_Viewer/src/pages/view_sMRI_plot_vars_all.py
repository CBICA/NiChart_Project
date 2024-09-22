import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.express as px
from math import ceil
from streamlit_plotly_events import plotly_events
from utils_trace import *
from st_pages import hide_pages
import os
import tkinter as tk
from tkinter import filedialog

def browse_file(path_input):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = path_input)
    path_out = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_out

def browse_folder(path_input):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = path_input)
    root.destroy()
    return out_path

#hide_pages(["Image Processing", "Data Analytics"])

def add_plot():
    '''
    Adds a new plot (updates a dataframe with plot ids)
    '''
    df_p = st.session_state.plots
    plot_id = f'Plot{st.session_state.plot_index}'
    df_p.loc[plot_id] = [plot_id, 
                         st.session_state.plot_xvar,
                         st.session_state.plot_yvar,
                         st.session_state.plot_hvar,
                         st.session_state.plot_trend
                        ]
    st.session_state.plot_index += 1

# Remove a plot
def remove_plot(plot_id):
    '''
    Removes the plot with the plot_id (updates the plot ids dataframe)
    '''
    df_p = st.session_state.plots
    df_p = df_p[df_p.PID != plot_id]
    st.session_state.plots = df_p

def display_plot(plot_id):
    '''
    Displays the plot with the plot_id
    '''

    def callback_plot_clicked():
        '''
        Set the active plot id to plot that was clicked
        '''
        st.session_state.plot_active = plot_id

    # Create a copy of dataframe for filtered data
    df_filt = df.copy()

    # Main container for the plot
    with st.container(border=True):

        with st.container(border=True):

            # Tabs for parameters
            ptabs = st.tabs([":lock:", ":large_orange_circle:", ":large_yellow_circle:",
                            ":large_green_circle:", ":x:"])

            # Tab 0: to hide other tabs

            # Tab 1: to set plotting parameters
            with ptabs[1]:
                plot_type = st.selectbox("Plot Type", ["DistPlot", "RegPlot"], key=f"plot_type_{plot_id}")
                
                # Get plot params
                xvar = st.session_state.plots.loc[plot_id].xvar
                yvar = st.session_state.plots.loc[plot_id].yvar
                hvar = st.session_state.plots.loc[plot_id].hvar
                trend = st.session_state.plots.loc[plot_id].trend

                # Select plot params from the user
                xind = df.columns.get_loc(xvar)
                yind = df.columns.get_loc(yvar)
                hind = df.columns.get_loc(hvar)
                tind = st.session_state.trend_types.index(trend)

                xvar = st.selectbox("X Var", df_filt.columns, 
                                     key=f"plot_xvar_{plot_id}", index = xind)
                yvar = st.selectbox("Y Var", df_filt.columns, 
                                     key=f"plot_yvar_{plot_id}", index = yind)
                hvar = st.selectbox("Hue Var", df_filt.columns, 
                                       key=f"plot_hvar_{plot_id}", index = hind)
                trend = st.selectbox("Trend Line", st.session_state.trend_types, 
                                          key=f"trend_type_{plot_id}", index = tind)

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
                cent_type = st.selectbox("Centile Type", ['CN-All', 'CN-F', 'CN-M'], key=f"cent_type_{plot_id}")

            # Tab 4: to reset parameters or to delete plot
            with ptabs[4]:
                st.button('Delete Plot', key=f'p_delete_{plot_id}',
                        on_click=remove_plot, args=[plot_id])

        # Main plot
        if trend == 'none':
            scatter_plot = px.scatter(df_filt, x = xvar, y = yvar, color = hvar)
        else:
            scatter_plot = px.scatter(df_filt, x = xvar, y = yvar, color = hvar, trendline = trend)

        # Add plot
        # - on_select: when clicked it will rerun and return the info

        sel_info = st.plotly_chart(scatter_plot, key=f"bubble_chart_{plot_id}", 
                                   on_select = callback_plot_clicked)

        # Detect MRID from the click info and save to session_state
        if len(sel_info['selection']['points'])>0:

            sind = sel_info['selection']['point_indices'][0]
            lgroup = sel_info['selection']['points'][0]['legendgroup']

            sel_mrid = df_filt[df_filt[hvar] == lgroup].iloc[sind]['MRID']
            sel_roi = st.session_state.plots.loc[st.session_state.plot_active, 'yvar']

            st.session_state.sel_mrid = sel_mrid
            st.session_state.sel_roi = sel_roi

            
            st.sidebar.success('Selected subject: ' +  sel_mrid)
            st.sidebar.success('Selected ROI: ' + sel_roi)
        # )
            


def filter_dataframe(df: pd.DataFrame, plot_id) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df_init = df.copy()
    df = df.copy()

    # Create filters selected by the user
    modification_container = st.container()
    with modification_container:
        widget_no = plot_id + '_filter'
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key = widget_no)
        for vno, column in enumerate(to_filter_columns):
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                widget_no = plot_id + '_col_' + str(vno)
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key = widget_no,
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


# Page controls in side bar
with st.sidebar:

    with st.container(border=True):

        # Input file name
        if st.sidebar.button("Select input file"):
            st.session_state.path_csv_mlscores, st.session_state.path_last_sel = browse_file(st.session_state.path_last_sel)
        csv_mlscores = st.sidebar.text_input("Enter the name of the ROI csv file:",
                                          value = st.session_state.path_csv_mlscores,
                                          label_visibility="collapsed")
        if os.path.exists(csv_mlscores):
            st.session_state.path_csv_mlscores = csv_mlscores

if os.path.exists(csv_mlscores):

    # Read input csv
    df = pd.read_csv(csv_mlscores)

    with st.sidebar:
        with st.container(border=True):

            # Slider to set number of plots in a row
            st.session_state.plot_per_raw = st.slider('Plots per row',1, 5, 3, key='a_per_page')

        with st.container(border=True):

            st.write('Plot Settings')

            # Tabs for parameters
            ptabs = st.tabs([":lock:", ":large_orange_circle:", ":large_yellow_circle:",
                            ":large_green_circle:"])

            # Tab 0: to set plotting parameters
            with ptabs[1]:
                # Default values for plot params
                st.session_state.plot_hvar = 'Sex'

                plot_xvar_ind = 0
                if st.session_state.plot_xvar in df.columns:
                    plot_xvar_ind = df.columns.get_loc(st.session_state.plot_xvar)

                plot_yvar_ind = 0
                if st.session_state.plot_yvar in df.columns:
                    plot_yvar_ind = df.columns.get_loc(st.session_state.plot_yvar)

                plot_hvar_ind = 0
                if st.session_state.plot_hvar in df.columns:
                    plot_hvar_ind = df.columns.get_loc(st.session_state.plot_hvar)

                st.session_state.plot_xvar = st.selectbox("Default X Var", df.columns, key=f"plot_xvar_init",
                                                            index = plot_xvar_ind)
                st.session_state.plot_yvar = st.selectbox("Default Y Var", df.columns, key=f"plot_yvar_init",
                                                            index = plot_yvar_ind)
                st.session_state.sel_var = st.session_state.plot_yvar

                st.session_state.plot_hvar = st.selectbox("Default Hue Var", df.columns, key=f"plot_hvar_init",
                                                                index = plot_hvar_ind)
                trend_index = st.session_state.trend_types.index(st.session_state.plot_trend)
                st.session_state.plot_trend = st.selectbox("Default Trend Line", st.session_state.trend_types,
                                                                key=f"trend_type_init", index = trend_index)

        # Button to add a new plot
        if st.button("Add plot"):
            add_plot()

    # Add a single plot (default: initial page displays a single plot)
    if st.session_state.plots.shape[0] == 0:
        add_plot()

    # Read plot ids
    df_p = st.session_state.plots
    list_plots = df_p.index.tolist()
    plot_per_raw = st.session_state.plot_per_raw

    # Render plots
    #  - iterates over plots;
    #  - for every "plot_per_raw" plots, creates a new columns block, resets column index, and displays the plot
    for i, plot_ind in enumerate(list_plots):
        column_no = i % plot_per_raw
        if column_no == 0:
            blocks = st.columns(plot_per_raw)
        with blocks[column_no]:
            display_plot(plot_ind)


    # FIXME: this is for debugging; will be removed
    with st.expander('session_state: Plots'):
        st.session_state.plot_active

    with st.expander('session_state: Plots'):
        st.session_state.plots

    with st.expander('session_state: All'):
        st.write(st.session_state)
