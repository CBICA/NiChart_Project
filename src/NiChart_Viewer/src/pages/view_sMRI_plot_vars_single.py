import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def calc_subject_centiles(df_subj: pd.DataFrame, df_cent: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate subject specific centile values
    """

    # Filter centiles to subject's age
    tmp_ind = (df_cent.Age - df_subj.Age[0]).abs().idxmin()
    sel_age = df_cent.loc[tmp_ind, "Age"]
    df_cent_sel = df_cent[df_cent.Age == sel_age]

    # Find ROIs in subj data that are included in the centiles file
    sel_vars = df_subj.columns[df_subj.columns.isin(df_cent_sel.ROI.unique())].tolist()
    df_cent_sel = df_cent_sel[df_cent_sel.ROI.isin(sel_vars)].drop(
        ["ROI", "Age"], axis=1
    )

    cent = df_cent_sel.columns.str.replace("centile_", "").astype(int).values
    vals_cent = df_cent_sel.values
    vals_subj = df_subj.loc[0, sel_vars]

    cent_subj = np.zeros(vals_subj.shape[0])
    for i, sval in enumerate(vals_subj):
        # Find nearest x values
        ind1 = np.where(vals_subj[i] < vals_cent[i, :])[0][0] - 1
        ind2 = ind1 + 1

        print(ind1)

        # Calculate slope
        slope = (cent[ind2] - cent[ind1]) / (vals_cent[i, ind2] - vals_cent[i, ind1])

        # Estimate subj centile
        cent_subj[i] = cent[ind1] + slope * (vals_subj[i] - vals_cent[i, ind1])

    df_out = pd.DataFrame(dict(ROI=sel_vars, Centiles=cent_subj))
    return df_out


def display_plot(sel_mrid: str) -> None:
    """
    Displays the plot with the given mrid
    """

    # Data columns
    dtmp = ["TotalBrain", "GM", "WM", "Ventricles"]
    vtmp = df[df.MRID == sel_mrid][dtmp].iloc[0].values.tolist()

    # Main container for the plot
    plot_container = st.container(border=True)
    with plot_container:
        # fig = px.bar(df, x='Fruit', y='Quantity', title='Fruit Consumption')
        fig = px.bar(x=dtmp, y=vtmp, title="Data Values")
        st.plotly_chart(fig)


# # Config page
# st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout='wide')

# FIXME: Input data is hardcoded here for now
fname = "../examples/test_input3/ROIS_tmp2.csv"
df = pd.read_csv(fname)

# Page controls in side bar
with st.sidebar:

    # Show selected id (while providing the user the option to select it from the list of all MRIDs)
    # - get the selected id from the session_state
    # - create a selectbox with all MRIDs
    # -- initialize it with the selected id if it's set
    # -- initialize it with the first id if not
    sel_mrid = st.session_state.sel_mrid
    if sel_mrid == "":
        sel_ind = 0
        sel_type = "(auto)"
    else:
        sel_ind = df.MRID.tolist().index(sel_mrid)
        sel_type = "(user)"
    sel_mrid = st.selectbox(
        "Select Subject", df.MRID.tolist(), key="selbox_mrid", index=sel_ind
    )

    # st.sidebar.warning('Selected subject: ' + mrid)
    st.success(f"Selected {sel_type}: {sel_mrid}")

    st.write("---")

display_plot(sel_mrid)
# # Button to add a new plot
# if st.button("Add plot"):
#     display_plot(sel_mrid)
