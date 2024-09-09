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

# # Initiate Session State Values
# if 'instantiated' not in st.session_state:
#     st.session_state.plots = pd.DataFrame({'PID':[]})
#     st.session_state.pid = 1
#     st.session_state.instantiated = True

# Config page
st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout='wide')

st.sidebar.image("../resources/nichart1.png")



with st.sidebar.expander("Acknowledgments"):
    st.markdown("""
                The CBICA Dev team
                """)

def display_plot(sel_id):
    '''
    Displays the plot with the given mrid
    '''

    # Create a copy of dataframe for filtered data
    df_sel = df[df.MRID == sel_id]

    # Data columns
    dtmp = ['TotalBrain', 'GM', 'WM', 'Ventricles']
    vtmp = df[df.MRID == sel_id][dtmp].iloc[0].values.tolist()

    # Main container for the plot
    plot_container = st.container(border=True)
    with plot_container:
        # fig = px.bar(df, x='Fruit', y='Quantity', title='Fruit Consumption')
        fig = px.bar(x=dtmp, y=vtmp, title='Data Values')
        st.plotly_chart(fig)

# FIXME: Input data is hardcoded here for now
fname = "../examples/test_input/vTest1/Study1/StudyTest1_DLMUSE_All.csv"
df = pd.read_csv(fname)

# Page controls in side bar
with st.sidebar:

    # Show selected id (while providing the user the option to select it from the list of all MRIDs)
    # - get the selected id from the session_state
    # - create a selectbox with all MRIDs
    # -- initialize it with the selected id if it's set
    # -- initialize it with the first id if not
    st.markdown("# Plotting single subject")
    st.markdown("""
                Info about this(idk)....
                """)

    sel_id = st.session_state.sel_id
    if sel_id == '':
        sel_ind = 0
        sel_type = '(auto)'
    else:
        sel_ind = df.MRID.tolist().index(sel_id)
        sel_type = '(user)'
    sel_id = st.selectbox("Select Subject", df.MRID.tolist(), key=f"selbox_mrid", index = sel_ind)

    # st.sidebar.warning('Selected subject: ' + mrid)
    st.warning(f'Selected {sel_type}: {sel_id}')

    st.write('---')

    st.sidebar.info("""
                    Note: This website is based on materials from the [NiChart Project](https://neuroimagingchart.com/).
                    The content and the logo of NiChart are intellectual property of [CBICA](https://www.med.upenn.edu/cbica/).
                    Make sure that you read the [licence](https://github.com/CBICA/NiChart_Project/blob/main/LICENSE).
                    """)

display_plot(sel_id)
    # # Button to add a new plot
    # if st.button("Add plot"):
    #     display_plot(sel_id)
