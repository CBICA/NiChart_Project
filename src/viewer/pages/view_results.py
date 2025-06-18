import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_io as utilio
import utils.utils_plots as utilpl
import utils.utils_mriview as utilmri
import utils.utils_data_view as utildv
import os
from pathlib import Path

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def panel_data_overview():
    '''
    Detect all csv files and merge them
    '''
    in_dir = st.session_state.paths['project']
    df_outdirs = st.session_state.df_outdirs

    utildv.data_overview(in_dir, df_outdirs)

def panel_data_merge():
    '''
    Detect all csv files and merge them
    '''
    in_dir = st.session_state.paths['project']
    key = 'MRID'
    df_outdirs = st.session_state.df_outdirs

    utildv.data_merge(in_dir, df_outdirs, key)


def plot_vars():
    """
    Panel for viewing dlmuse results
    """    
    # Select result type        
    list_res_type = ['Segmentation', 'Volumes']
    sel_res_type = st.pills(
        'Select output type',
        list_res_type,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        
        utilmri.panel_view_seg(ulay, olay, 'muse')
        
    elif sel_res_type == 'Volumes':
        # df = pd.read_csv(
        #     '/home/guraylab/GitHub/gurayerus/NiChart_Project/resources/reference_data/centiles/dlmuse_centiles_CN.csv'
        #     #'/home/gurayerus/GitHub/gurayerus/NiChart_Project/resources/reference_data/centiles/dlmuse_centiles_CN.csv'
        # )
        st.session_state.curr_df = None
        utilpl.panel_view_centiles('dlmuse', 'rois')
         
    print(st.session_state.selections)
    #print(st.session_state.plot_params)

def view_images():
    st.write('wait')

st.markdown(
    """
    ### View Results 
    """
)

my_tabs = st.tabs(
    ["Overview", "Merge Variables", "Plot Variables", "View Images"]
)

with my_tabs[0]:
    panel_data_overview()

with my_tabs[1]:
    panel_data_merge()

with my_tabs[2]:
    plot_vars()

with my_tabs[3]:
    view_images()
