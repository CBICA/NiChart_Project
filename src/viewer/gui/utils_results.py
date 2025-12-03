import os
import shutil
import time
from typing import Any

import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import utils.utils_misc as utilmisc
import utils.utils_user_select as utiluser
import utils.utils_io as utilio

import utils.utils_session as utilses
import gui.utils_plots as utilpl
import gui.utils_mriview as utilmri
import gui.utils_view as utilview
import pandas as pd
import gui.utils_widgets as utilwd

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

def set_plot_params():
    """
    Panel for selecting plotting parameters
    """

    ###################################################
    ### Set specific selections for different pipelines
    pipeline = st.session_state.general_params['sel_pipeline']

    st.session_state.plot_params['xvargroup'] = 'age'
    st.session_state.plot_params['xvar'] = 'Age'

    if pipeline == 'dlmuse':
        yvarlist = ['roi']
        st.session_state.plot_params['yvargroup'] = 'MUSE_ShortList'
        st.session_state.plot_params['yvar'] = 'GM'

    elif pipeline == 'dlwmls':
        yvarlist = ['wmroi']
        st.session_state.plot_params['yvargroup'] = 'MUSE_WM'
        st.session_state.plot_params['yvar'] = 'Frontal_WM_R'

    elif pipeline == 'spare':
        yvarlist = ['biomarker']
        st.session_state.plot_params['yvargroup'] = 'SPARE_Scores'
        st.session_state.plot_params['yvar'] = 'SPARE_BA'


    sac.divider(label='Plotting Parameters', align='center', color='indigo', size='lg')
    
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Variables'),
            sac.TabsItem(label='Centiles'),
            sac.TabsItem(label='Filters'),
            sac.TabsItem(label='Trends'),
            sac.TabsItem(label='Plot Settings'),
        ],
        size='sm',
        align='left'
    )
    
    
    #### Variables
    if tab == 'Variables':
        sel_xvar = utilwd.select_var_twolevels(
            'plot_params', 'xvargroup', 'xvar',
            'Variable X', ['age'],
        )
        
        sel_yvar = utilwd.select_var_twolevels(
            'plot_params', 'yvargroup', 'yvar',
            'Variable Y', yvarlist
        )

        sel_hvar = utilwd.select_var_twolevels(
            'plot_params', 'hvargroup', 'hvar',
            'Grouping Variable', ['demog']
        )
        
    #### Centiles
    if tab == 'Centiles':
        sac.divider(label='Centiles', align='center', color='indigo', size='lg')
        utilwd.select_centiles()        

    #### Filters
    if tab == 'Filters':

        # Let user select sex var
        sel_sex = utilwd.my_multiselect('plot_params', 'filter_sex', ['F','M'], 'Sex')

        # Let user pick an age range
        sel_age_range = utilwd.my_slider(
            'plot_params', 'filter_age', 'Age Range', 0, 110
        )

    #### Trends
    if tab == 'Trends':
        sac.divider(label='Trends', align='center', color='indigo', size='lg')
        utilwd.select_trend()
    

    #### Plot Settings
    if tab == 'Plot Settings':
        utilwd.select_plot_settings()

def set_plot_controls():
    sac.divider(label='Plot Controls', align='center', color='indigo', size='lg')
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button('Add Plot'):
            st.session_state.plots = utilpl.add_plot(
                st.session_state.plots, st.session_state.plot_params
            )
            #st.write(st.session_state.plots)
            #st.write(st.session_state.plot_params)
        if st.button('Delete Selected'):
            st.session_state.plots = utilpl.delete_sel_plots(
                st.session_state.plots
            )

        if st.button('Delete All'):
            st.session_state.plots = utilpl.delete_all_plots()


def plot_data(layout):
    """
    View img variables
    """
    with layout:
        set_plot_params()

    with layout:
        set_plot_controls()

    # Update traces
    plot_params = st.session_state.plot_params
    
    plot_params['traces'] = ['data']
    if plot_params['centile_values'] is not None:
        if st.session_state.plot_data['df_cent'] is None:
            st.warning('Note: Reference centile data is not available!')
        else:
            plot_params['traces'] = plot_params['traces'] + plot_params['centile_values']

    if plot_params['trend'] == 'Linear':
        plot_params['traces'] = plot_params['traces'] + ['lin_fit']

    if plot_params['show_conf']:
        plot_params['traces'] = plot_params['traces'] + ['conf_95%']

    if plot_params['trend'] == 'Smooth LOWESS Curve':
        plot_params['traces'] = plot_params['traces'] + ['lowess']

    #st.write(st.session_state.plot_data)

    utilpl.panel_show_plots()
    
def view_segmentation(layout):
    """
    View segmentations
    """
    pipeline = st.session_state.general_params['sel_pipeline']

    with layout:
        sac.divider(label='Data', align='center', color='grape', size = 'xl')

    if pipeline == 'dlmuse':
        fname = os.path.join(
            st.session_state.paths['curr_data'], 'dlmuse_vol', 'DLMUSE_Volumes.csv'
        )
        df = pd.read_csv(fname)
        df.columns = df.columns.str.replace('DL_MUSE_Volume_','')
        df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])
        list_vars = df.columns.unique().tolist()
        list_mrids = df.MRID.sort_values().tolist()
        
        with layout:
            sel_mrid = utilwd.my_selectbox(
                'mriplot_params', 'sel_mrid', list_mrids, 'Subject'
            )
        if sel_mrid is None or sel_mrid == 'Select an option…':
            return

        #######################
        ## Set olay ulay images
        fname = os.path.join(
            st.session_state.paths['curr_data'], 't1', f'{sel_mrid}_T1.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['ulay'] = None
            st.write(fname)
        else:
            st.session_state.mriplot_params['ulay'] = fname

        fname = os.path.join(
            st.session_state.paths['curr_data'], 'dlmuse_seg', f'{sel_mrid}_T1_DLMUSE.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['olay'] = None
            st.write(fname)
        else:
            st.session_state.mriplot_params['olay'] = fname
            
        # Select ROI
        with layout:
            sel_roi = utilwd.select_muse_roi(list_vars)
        if sel_roi is None or sel_roi == 'Select an option…':
            return
        st.session_state.mriplot_params['sel_roi'] = sel_roi

        # Select plot parameters
        with layout:
            utilwd.select_mriplot_settings()
            
        if st.session_state.workflow == 'ref_data':
            st.warning('**Note:** This is a low-resolution (2 mm) sample dataset provided for illustration only.')
        
        utilmri.panel_view_seg()

    elif pipeline == 'dlwmls':
        fname = os.path.join(
            st.session_state.paths['curr_data'], 'nichart_dlwmls_out', 'DLWMLS_DLMUSE_Segmented_Volumes.csv'
        )
        df = pd.read_csv(fname)
        df.columns = df.columns.str.replace('DL_WMLS_Volume_','')
        list_mrids = df.MRID.sort_values().tolist()
        
        with layout:
            sel_mrid = utilwd.my_selectbox(
                'mriplot_params', 'sel_mrid', list_mrids, 'Subject'
            )
        if sel_mrid is None or sel_mrid == 'Select an option…':
            return

        #######################
        ## Set olay ulay images
        fname = os.path.join(
            st.session_state.paths['curr_data'], 'fl', f'{sel_mrid}_FL.nii.gz'
        )
        # FIXME
        fname = os.path.join(
            st.session_state.paths['curr_data'], 't1', f'{sel_mrid}_T1.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['ulay'] = None
            st.write(fname)
        else:
            st.session_state.mriplot_params['ulay'] = fname

        fname = os.path.join(
            st.session_state.paths['curr_data'], 'nichart_dlwmls_out', 
            'DLWMLS_DLMUSE_Segmented',
            f'{sel_mrid}_DLWMLS_DLMUSE_Segmented.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['olay'] = None
            st.write(fname)
        else:
            st.session_state.mriplot_params['olay'] = fname
            
        st.session_state.mriplot_params['sel_roi'] = None

        # Select plot parameters
        with layout:
            utilwd.select_mriplot_settings()
            
        if st.session_state.workflow == 'ref_data':
            st.warning('**Note:** This is a low-resolution (2 mm) sample dataset provided for illustration only.')
        
        utilmri.panel_view_seg()


def prep_csv():
    """
    Merge result files to view
    """
    pipeline = st.session_state.general_params['sel_pipeline']

    out_dir = os.path.join(
        st.session_state.paths['curr_data'], 'plots'
    )
    fout = os.path.join(
        out_dir, f'data_{pipeline}.csv'
    )
    os.makedirs(out_dir, exist_ok=True)

    f_p = os.path.join(
        st.session_state.paths['curr_data'], 'participants', 'participants.csv'
    )

    # Set pipeline specific parameters    
    if pipeline == 'dlmuse':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'dlmuse_vol', 'DLMUSE_Volumes.csv'
        )

    elif pipeline == 'dlwmls':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'nichart_dlwmls_out', 'DLWMLS_DLMUSE_Segmented_Volumes.csv'
        )

    elif pipeline == 'spare':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'ml_biomarkers', 'SPARE_ALL.csv'
        )
    
    try:
        df_p = pd.read_csv(f_p)
        df_d = pd.read_csv(f_d)
        df = df_p.merge(df_d, on='MRID')
        df.to_csv(fout, index=False)
        st.toast('Data file merged to participant info!')
    except:
        st.warning('Could not read result files!')

def view_img_vars(layout):
    """
    View image variables
    """
    pipeline = st.session_state.general_params['sel_pipeline']
    if str(pipeline) == 'Select an option...':
        return
    
    # Set reference centile data
    fname = os.path.join(
        st.session_state.paths['centiles'],
        pipeline + '_centiles_' + st.session_state.plot_params['centile_type'] + '.csv'
    )
    if fname != st.session_state.plot_data['csv_cent']:
        st.session_state.plot_data['csv_cent'] = fname
        try:
            df = utilio.read_csv(fname)
            st.session_state.plot_data['df_cent'] = df
        except:
            st.session_state.plot_data['df_cent'] = None


    # Set data file
    fname = os.path.join(st.session_state.paths['curr_data'], 'plots', f'data_{pipeline}.csv')
    
    if fname != st.session_state.plot_data['csv_data']:
        prep_csv()

        st.session_state.plot_data['csv_data'] = fname
        df = utilio.read_csv(fname)
        
        # Pipeline specific steps
        if pipeline == 'dlmuse':            
            df.columns = df.columns.str.replace('DL_MUSE_Volume_','')
            df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])

        if pipeline == 'dlwmls':            
            df.columns = df.columns.str.replace('DL_WMLS_Volume_','')
            df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])
            #st.write(df)
            
        if pipeline == 'spare':            
            df = df.drop('SPARE_AD', axis=1)
            df.columns = df.columns.str.replace('SPARE_AD_decision_function','SPARE_AD')
            #st.write(df)
            
        df["grouping_var"] = "Data"
        st.session_state.plot_data['df_data'] = df

    ## Pipeline specific steps
    #if pipeline == 'dlmuse':

    ##elif pipeline == 'spare':

    ##else:
    ###elif pipeline == 'dlwmls':
        ##st.info('Not available yet ...')
        ##return

    # Plot data
    plot_data(layout)

def prepare_data_for_download(prj_dir, sel_opt, out_zip):
    utilio.zip_folders(prj_dir, sel_opt, out_zip)
    with open(out_zip, "rb") as f:
        file_download = f.read()
    st.toast('Created zip file with selected folders')
    os.remove(out_zip)
    return file_download

def panel_info():
    with st.container(border=True):
        st.markdown(
            '''
            - NiChart Reference Dataset is a large and diverse collection from multiple MRI studies, created as part of the ISTAGING project to develop a system for identifying imaging biomarkers of aging and neurodegenerative diseases.

            - The dataset includes multi-modal MRI data, as well as carefully curated demographic, clinical, and cognitive variables from participants with a variety of health conditions.

            - The reference dataset is used for training machine learning models and for creating reference distributions of imaging measures and signatures

            - Users can compare their values to normative or disease-related reference distributions.            '''
        )
        st.image(
            os.path.join(
                st.session_state.paths['resources'], 'images', 'nichart_data.png'
            ),
            width=1200
        )

def panel_download():
    '''
    Panel to download results
    '''
    if st.session_state.workflow == 'ref_data':
        st.info('Reference data download is not available at this time.')
        return
    
    with st.container(horizontal=True, horizontal_alignment="center"):

        st.markdown(f"###### 📁 Project Folder:   `{st.session_state.prj_name}`", width='content')
    
        prj_dir = st.session_state.paths['prj_dir']
        list_dirs = utilio.get_subfolders(prj_dir)
        for folder_name in ['download', 'downloads', 'user_upload']:
            if folder_name in list_dirs:
                list_dirs.remove(folder_name)
        
        if len(list_dirs) == 0:
            return
        
        sel_opt = sac.checkbox(
            list_dirs,
            label='Folder(s) to download:', align='center', 
            color='#aaeeaa', size='xl',
            check_all='Select all'
        )

        with st.container(horizontal=True, horizontal_alignment="center"):
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

def panel_results():
    logger.debug('    Function: panel_results')

    if st.session_state.workflow is None:
        st.info('Please select a Workflow!')
        return

    # Set plotting parameters layout
    if st.session_state.layout_plots == 'Main':
        layout = st.container(border=False)
    else:
        layout = st.sidebar

    with layout:
        sac.divider(label='General Options', align='center', color='indigo', size='lg')

    with layout:
        if st.session_state.workflow == 'ref_data':
            sel_task = utilwd.my_selectbox(
                'general_params', 'sel_task', ['Info', 'View'], hdr='Task'
            )
        else:
            sel_task = utilwd.my_selectbox(
                'general_params', 'sel_task', ['Download', 'View'], hdr='Task'
            )

    if sel_task == 'Info':
        panel_info()

    elif sel_task == 'Download':
        panel_download()

    elif sel_task == 'View':
        with layout:
            sel_rtype = utilwd.my_selectbox(
                'general_params', 'sel_rtype',
                ['Numeric', 'Image'], 'Data Type'
            )

        if sel_rtype == 'Image':
            with layout:
                sel_pipe = utilwd.my_selectbox(
                    'general_params', 'sel_pipeline',
                    ['dlmuse', 'dlwmls'], 'Pipeline'
                )
            if sel_pipe is None or sel_pipe == 'Select an option...':
                return
            view_segmentation(layout)

        elif sel_rtype == 'Numeric':
            with layout:
                old_pipe = st.session_state.general_params['sel_pipeline']
                sel_pipe = utilwd.my_selectbox(
                    'general_params', 'sel_pipeline',
                    ['dlmuse', 'dlwmls', 'spare', 'cclnmf', 's-gan'], 'Pipeline'
                )
                
            #st.write(sel_pipe)
            print(sel_pipe)
            
            if sel_pipe is None or str(sel_pipe) == 'Select an option...':
                return
            #else:
                #st.write(sel_pipe)
                #print(f'aaaaa {sel_pipe} Select an option...')
                #st.write(str(sel_pipe) == 'Select an option...')
                
            #return
            
            # Reset plots if pipeline changed
            if old_pipe != sel_pipe:
                st.session_state.plots = pd.DataFrame(columns=['flag_sel', 'params'])
                st.session_state.plot_curr = -1
                st.session_state.plot_active = None
                
            view_img_vars(layout)


