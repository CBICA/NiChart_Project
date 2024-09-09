import pandas as pd
import streamlit as st
from io import StringIO
from typing import Any
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from typing import Any
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="DataFrame Demo", page_icon="📊")
st.markdown("# CSV data visualization")
st.sidebar.header("CSV data visualization")

st.write(
    """
        View NiChart imaging variables and biomarkers
    """
)
st.sidebar.image("../resources/nichart1.png")

with st.sidebar:
    st.markdown("# How to upload input files")
    st.markdown("""
                You can upload 1 or more csv files in once, the viewer \
                will provide a dropdown selection for you to choose which \
                csv file you want to work with. \
                """)

with st.sidebar:
    st.markdown("# DataFrame filtering")
    st.markdown("""
                The filtering is done to a subset of the csv file as the visualization \
                for such big datasets is not efficient. Have that in mind while you study the \
                charts. \
                """)

st.sidebar.info("""
                    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
                    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
                    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
                    """)

with st.sidebar.expander("Acknowledgments"):
    st.markdown("""
                The CBICA Dev team
                """)



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
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

    return df


def df_piechart(df: pd.DataFrame) -> Any:
    # Let user select the column for the pie chart
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    selection = st.selectbox("Select column for pie chart", df.columns)

    # Group by the selected column to get counts for the pie chart
    counts = df[selection].value_counts()

    # Create a pie chart using matplotlib
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.set_title(f"Piechart of {selection}")
    return fig

def df_histchart(df: pd.DataFrame) -> Any:
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    selection = st.selectbox("Select column for histogram", numeric_columns)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(df[selection], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(f"Histogram of {selection}")
    ax.set_xlabel(selection)
    ax.set_ylabel("Frequency")
    return fig

uploaded_file = st.file_uploader("Upload input csv files", accept_multiple_files=True)

df = None
if len(uploaded_file) != 0:
    csv_files = []
    csv_files_hash = {}
    for (i, csv) in enumerate(uploaded_file):
        df = pd.read_csv(csv)
        os.makedirs("user_input/", exist_ok=True)
        filepath = os.path.join("user_input/", csv.name)
        df.to_csv(filepath, header=True, index=False)
        csv_files.append(csv.name)
        csv_files_hash[csv.name] = df

    csv_select = st.selectbox("Select dataset", csv_files)
    st.write(f"Selected dataframe: {csv_select}")
    st.write(f"Data Shape: {csv_files_hash[csv_select].shape}")

    df_reduced = csv_files_hash[csv_select].head(40)
    st.dataframe(filter_dataframe(df_reduced))

    piechart = df_piechart(df_reduced)
    st.pyplot(piechart)

    histchart = df_histchart(df_reduced)
    st.pyplot(histchart)
