import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

if 'flag' not in st.session_state:
    st.session_state['flag'] = None

def toggle_add_plot():
    print('toggled')
    st.session_state['flag'] = st.session_state['_tmp']
    st.session_state['_tmp'] = None


options = ['Add Plot', 'Delete Selected', 'Delete All']

sel = st.segmented_control(
    "Directions",
    options,
    selection_mode="single",
    on_change = toggle_add_plot,
    key = '_tmp'
)
st.markdown(f"Your selected options: {sel}.")

if st.session_state['flag'] == 'Add Plot':
    st.write('add')

if st.session_state['flag'] == 'Delete Selected':
    st.write('delete')

with st.container(border=True):
    st.write(st.session_state)
