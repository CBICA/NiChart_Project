import streamlit as st
import utils.utils_io as utilio
import os

def prepare_data_for_download(prj_dir, sel_opt, out_zip):
    utilio.zip_folders(prj_dir, sel_opt, out_zip)
    with open(out_zip, "rb") as f:
        file_download = f.read()
    st.toast('Created zip file with selected folders')
    os.remove(out_zip)
    return file_download

def panel_download():
    '''
    Panel to download results
    '''
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown(
            f":violet-badge[Current Project] :green-badge[:material/folder: {st.session_state.prj_name}]"
        )

    with st.container(horizontal=False, horizontal_alignment="left", border=True):

        st.markdown('**Folder(s) to download:**')

        prj_dir = st.session_state.paths['prj_dir']
        list_dirs = utilio.get_subfolders(prj_dir)
        for folder_name in ['download', 'downloads', 'user_upload']:
            if folder_name in list_dirs:
                list_dirs.remove(folder_name)
        
        if len(list_dirs) == 0:
            return
        
        select_all = st.checkbox('Select all')
        selected = st.pills(
            'Folder(s) to download:',
            list_dirs,
            selection_mode="multi",
            label_visibility='collapsed',
            disabled=select_all
        )
        sel_opt = list_dirs if select_all else (selected or [])

    with st.container(horizontal=True, horizontal_alignment="left", border=False):
        flag_disabled1 = True
        flag_disabled2 = True
        data_zip = ''
        if sel_opt is not None and len(sel_opt)>0:
            flag_disabled1 = False
        
        if st.button('Prepare Data', disabled = flag_disabled1):
            out_dir = os.path.join(prj_dir, 'downloads')
            os.makedirs(out_dir, exist_ok=True)
            out_zip = os.path.join(out_dir, 'nichart_results.zip')
            data_zip = prepare_data_for_download(prj_dir, sel_opt, out_zip)
            flag_disabled2 = False

        #st.download_button(
            #label = f"Download",
            #data = prepare_data_for_download(prj_dir, sel_opt, out_zip),
            #file_name = 'nichart_results.zip',
            #on_click = 'ignore'
        #)

        st.download_button(
            label = f"Download",
            data = data_zip,
            file_name = 'nichart_results.zip',
            disabled = flag_disabled2
        )