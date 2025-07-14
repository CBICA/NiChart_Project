import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilses
import utils.utils_io as utilio
import streamlit_antd_components as sac
import os
from utils.utils_logger import setup_logger

logger = setup_logger()
logger.debug('Page: Download Results')



# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def download_folder(dtype):
    """
    Downloading files in folder
    """
    # Zip results and download
    out_zip = bytes()
    out_dir = os.path.join(
        st.session_state.paths['project'], 'download'
    )
    in_dir = os.path.join(
        st.session_state.paths['project'], dtype
    )
    if not os.path.exists(in_dir):
        st.error("Input data missing!")
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #try:
    f_tmp = os.path.join(out_dir, dtype)
    out_zip = utilio.zip_folder(in_dir, f_tmp)
    st.download_button(
        f"Download results: {dtype}",
        out_zip,
        file_name=f"{st.session_state.project}_{dtype}.zip",
    )
    #except:
        #st.error("Could not download data!")

def panel_download():
    '''
    Panel to download results
    '''
    prjdir = st.session_state.paths['project']
    list_dirs = utilio.get_subfolders(prjdir)
    if 'download' in list_dirs:
        list_dirs.remove('download')
    
    if len(list_dirs) == 0:
        return
    
    sel_dir = sac.segmented(
        items=list_dirs,
        size='sm',
        align='left'
    )

    if sel_dir is None:
        return
    
    #if st.button('Download'):
    download_folder(sel_dir)
    



st.markdown(
    """
    ### Download Results 
    """
)

panel_download()



# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
