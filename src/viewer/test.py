import streamlit as st
import pandas as pd

with st.container(border=True):

    # Get df columns
    df = pd.read_csv('/home/gurayerus/Desktop/DDD/rois.csv')

    vType = df.Type.unique()

    xvar = st.selectbox("X Var", vType, key="plot_x1", index=None)

    if xvar is None:
        vType2 = []
    else:
        vType2 = df[df.Type == xvar].Name.tolist()
    xvar2 = st.selectbox("X2 Var", vType2, key="plot_x2", index=None)

