import os
import shutil
import time
from typing import Any
import plotly.graph_objs as go

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

from NiChart_common_utils.nifti_parser import NiftiMRIDParser

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()


def _add_plots_for_group():
    """
    Add one plot for every variable in the currently visible tag-filtered group.
    Uses the same plot_params but iterates over each variable in the group.
    """
    df_vars = st.session_state.dicts['df_var_groups']

    # Respect the active tag filters stored by select_var_with_tag_filter
    pipe_key = '_tag_pipe_yvar'
    cat_key  = '_tag_cat_yvar'
    sel_pipes = st.session_state.get(pipe_key, [])
    sel_cats  = st.session_state.get(cat_key, [])

    if sel_pipes:
        def _pipe_match(pipes):
            return len(pipes) == 0 or bool(set(pipes) & set(sel_pipes))
        df_vars = df_vars[df_vars['pipeline'].apply(_pipe_match)]
    if sel_cats:
        df_vars = df_vars[df_vars['category'].isin(sel_cats)]

    # Filter to the currently selected group (first level)
    cur_group = st.session_state.plot_params.get('yvargroup')
    if cur_group and cur_group != 'Select an option...' and cur_group in df_vars['group'].values:
        df_vars = df_vars[df_vars['group'] == cur_group]

    # Resolve variable names (handle atlas-indexed groups)
    muse_dict = st.session_state.dicts['muse']['ind_to_name']
    for _, row in df_vars.iterrows():
        raw_vals = row['values']
        if row['atlas'] == 'muse':
            var_names = [muse_dict.get(v, v) for v in raw_vals]
        else:
            var_names = raw_vals

        for vname in var_names:
            params = st.session_state.plot_params.copy()
            params['yvargroup'] = row['group']
            params['yvar'] = vname
            st.session_state.plots = utilpl.add_plot(st.session_state.plots, params)


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
        st.session_state.plot_params['yvargroup'] = 'MUSE_ShortList'
        st.session_state.plot_params['yvar'] = 'GM'

    elif pipeline == 'dlwmls':
        st.session_state.plot_params['yvargroup'] = 'MUSE_WM'
        st.session_state.plot_params['yvar'] = 'Frontal_WM_R'

    elif pipeline == 'spare':
        st.session_state.plot_params['yvargroup'] = 'SPARE_Scores'
        st.session_state.plot_params['yvar'] = 'SPARE_BA'

    elif pipeline == 'spare_cvm':
        st.session_state.plot_params['yvargroup'] = 'SPARE_CVM_Scores'
        st.session_state.plot_params['yvar'] = 'SPARE_HYPERTENSION'

    elif pipeline == 'cclnmf':
        st.session_state.plot_params['yvargroup'] = 'CCLNMF_Aging_Dimensions'
        st.session_state.plot_params['yvar'] = 'CCL-NMF1'

    elif pipeline == 'surreal_gan':
        st.session_state.plot_params['yvargroup'] = 'SurrealGAN_Aging_Dimensions'
        st.session_state.plot_params['yvar'] = 'R1'

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

        sel_yvar = utilwd.select_var_with_tag_filter(
            'plot_params', 'yvargroup', 'yvar',
            'Variable Y',
        )

        sel_hvar = utilwd.select_var_twolevels(
            'plot_params', 'hvargroup', 'hvar',
            'Grouping Variable', ['cat_vars']
        )

        st.divider()
        with st.container(horizontal=True, horizontal_alignment='left'):
            if st.button('Add Plot', key='_add_plot_vars'):
                st.session_state.plots = utilpl.add_plot(
                    st.session_state.plots, st.session_state.plot_params
                )
            if st.button('Add for All in Group', key='_add_all_vars'):
                _add_plots_for_group()
        
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
        if sel_mrid is None or str(sel_mrid) == 'Select an option...':
            return

        #######################
        ## Set olay ulay images

        # Use heuristic parser
        # mod_dirs = {mod: os.path.join(st.session_state.paths['project'], mod) for mod in ['t1', 't2', 'fl', 'dti', 'fmri']}
        # dir_dict = {'T1': mod_dirs['t1'],
        #                         'T2': mod_dirs['t2'],
        #                         'FLAIR': mod_dirs['fl'],
        #                         'DTI': mod_dirs['dti'],
        #                         'FMRI': mod_dirs['fmri'],
        #                         'DLMUSE': os.path.join(st.session_state.paths['project'], 'dlmuse_seg')
        #                         }
        
        #nifti_parser = NiftiMRIDParser()
        #heuristic_df = nifti_parser.create_master_csv(dir_dict, os.path.join(st.session_state.paths['project'], 'inferred_data_paths.csv'))
        
        #heuristic_df = heuristic_df.sort_values(by='MRID')
        #fname = nifti_parser.get_path(sel_mrid, modality='t1')

        fname = os.path.join(
            st.session_state.paths['curr_data'], 't1', f'{sel_mrid}_T1.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['ulay'] = None
            st.write(fname)
        else:
            st.session_state.mriplot_params['ulay'] = fname

        #fname = nifti_parser.get_path(sel_mrid, modality='DLMUSE')
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
        if sel_roi is None or str(sel_roi) == 'Select an option...':
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
            st.session_state.paths['curr_data'], 'participants', 'participants.csv'
        )
        try: 
            df = pd.read_csv(fname)
            list_mrids = df.MRID.sort_values().tolist()
        except:
            st.warning('Could not detect result files for this pipeline!')
            return
        
        with layout:
            sel_mrid = utilwd.my_selectbox(
                'mriplot_params', 'sel_mrid', list_mrids, 'Subject'
            )
        if sel_mrid is None or str(sel_mrid) == 'Select an option...':
            return

        #######################
        ## Set olay ulay images
        fname = os.path.join(
            st.session_state.paths['curr_data'], 'fl', f'{sel_mrid}_FL.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['ulay'] = None
            st.warning('Could not detect underlay image!')
            return
        else:
            st.session_state.mriplot_params['ulay'] = fname

        fname = os.path.join(
            st.session_state.paths['curr_data'], 'nichart_dlwmls_out', 
            'DLWMLS_FLAIR',
            f'{sel_mrid}_FL_DLWMLS.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['olay'] = None
            st.warning('Could not detect overlay image!')
            return
        else:
            st.session_state.mriplot_params['olay'] = fname
            
        st.session_state.mriplot_params['sel_roi'] = None

        # Select plot parameters
        with layout:
            utilwd.select_mriplot_settings()
            
        if st.session_state.workflow == 'ref_data':
            st.warning('**Note:** This is a low-resolution (2 mm) sample dataset provided for illustration only.')
        
        utilmri.panel_view_seg()

    elif pipeline == 'csf_ravens':
        fname = os.path.join(
            st.session_state.paths['curr_data'], 'participants', 'participants.csv'
        )
        try: 
            df = pd.read_csv(fname)
            list_mrids = df.MRID.sort_values().tolist()
        except:
            st.warning('Could not detect result files for this pipeline!')
            return
        
        with layout:
            sel_mrid = utilwd.my_selectbox(
                'mriplot_params', 'sel_mrid', list_mrids, 'Subject'
            )
        if sel_mrid is None or str(sel_mrid) == 'Select an option...':
            return

        #######################
        ## Set olay ulay images
        fname = os.path.join(
            st.session_state.paths['curr_data'], 't1', f'{sel_mrid}_T1.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['ulay'] = None
            st.warning('Could not detect underlay image!')
            return
        else:
            st.session_state.mriplot_params['ulay'] = fname

        fname = os.path.join(
            st.session_state.paths['curr_data'], 'nichart_ravens_out', 
            f'{sel_mrid}_Label_CSF_RAVENS_ICVNorm_zScored_inSubj.nii.gz'
        )
        if not os.path.exists(fname):
            st.session_state.mriplot_params['olay'] = None
            st.warning('Could not detect overlay image!')
            return
        else:
            st.session_state.mriplot_params['olay'] = fname
            
        st.session_state.mriplot_params['sel_roi'] = None

        # Select plot parameters
        with layout:
            utilwd.select_ravensplot_settings()
            
        if st.session_state.workflow == 'ref_data':
            st.warning('**Note:** This is a low-resolution (2 mm) sample dataset provided for illustration only.')
        
        utilmri.panel_view_map()


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

    elif pipeline == 'spare_cvm':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'ml_biomarkers', 'SPARE_CVM_ALL.csv'
        )

    elif pipeline == 'cclnmf':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'ml_biomarkers', 'CCLNMF.csv'
        )

    elif pipeline == 'surreal_gan':
        f_d = os.path.join(
            st.session_state.paths['curr_data'], 'ml_biomarkers', 'SurrealGAN_RScores.csv'
        )
    
    try:
        df_p = pd.read_csv(f_p)
        df_d = pd.read_csv(f_d)
        df = df_p.merge(df_d, on='MRID')
        df.to_csv(fout, index=False)
        st.toast('Data file merged to participant info!')

    except:
        st.warning('Could not read result files!')
        return False

    return True

def rename_columns(df, suffix):
    tmp_cols = [c for c in df.columns if c.endswith(suffix)]
    
    for c in tmp_cols:
        base = c.replace(suffix, "")
        df[base] = df[c]

    df = df.drop(columns=tmp_cols)
    return df

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
        
        if not prep_csv():
            return

        st.session_state.plot_data['csv_data'] = fname
        df = utilio.read_csv(fname)
        
        # Pipeline specific steps
        if pipeline == 'dlmuse':            
            df.columns = df.columns.str.replace('DL_MUSE_Volume_','')
            df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])

        elif pipeline == 'dlwmls':            
            df.columns = df.columns.str.replace('DL_WMLS_Volume_','')
            df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])
            #st.write(df)
            
        elif pipeline == 'spare':            
            df = rename_columns(df, '_decision_function')
            #st.write(df)
            
        elif pipeline == 'spare_cvm':            
            df = rename_columns(df, '_decision_function')

        elif pipeline == 'surreal_gan':         
            df.columns = df.columns.str.replace('SurrealGAN_','')

        df["grouping_var"] = "Data"
        st.session_state.plot_data['df_data'] = df

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
    with st.container(horizontal=True, horizontal_alignment="center"):

        st.markdown(
            f":violet-badge[Current Project] :green-badge[:material/folder: {st.session_state.prj_name}]"
        )
    
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

    # Set plotting parameters layout
    if st.session_state.layout_plots == 'Main':
        layout = st.container(border=False)
    else:
        layout = st.sidebar

    with layout:
        sac.divider(label='General Options', align='center', color='indigo', size='lg')

    with layout:
        old_pipe = st.session_state.general_params['sel_pipeline']
        sel_pipe = utilwd.my_selectbox(
            'general_params', 'sel_pipeline',
            ['dlmuse', 'dlwmls', 'spare', 'spare_cvm', 'cclnmf', 'surreal_gan'],
            'Pipeline'
        )

    if sel_pipe is None or str(sel_pipe) == 'Select an option...':
        return
        
    # Reset plots if pipeline changed
    if old_pipe != sel_pipe:
        st.session_state.plots = pd.DataFrame(columns=['flag_sel', 'params'])
        st.session_state.plot_curr = -1
        st.session_state.plot_active = None
        
    view_img_vars(layout)


CENTILE_FILES = {
    'All (CN)': 'dlmuse_centiles_CN.csv',
    'Males (CN)': 'dlmuse_centiles_CN-Males.csv',
    'Females (CN)': 'dlmuse_centiles_CN-Females.csv',
    'ICV Normalized': 'dlmuse_centiles_CN-ICVNorm.csv',
}

def panel_centile_view():
    """
    Panel showing centile age trends for a selected variable.
    Controls (reference group + variable) on the left; centile band chart on the right.
    """
    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        sel_group = st.selectbox('Reference group', list(CENTILE_FILES.keys()))

        centile_path = os.path.join(
            st.session_state.paths['centiles'], CENTILE_FILES[sel_group]
        )
        try:
            df_cent = pd.read_csv(centile_path)
        except Exception:
            st.error('Could not load centile data.')
            return

        vars_list = sorted(df_cent['VarName'].unique().tolist())
        default = 'TotalBrain' if 'TotalBrain' in vars_list else vars_list[0]
        sel_var = st.selectbox('Variable', vars_list, index=vars_list.index(default))

    with col_plot:
        df_v = df_cent[df_cent['VarName'] == sel_var].sort_values('Age')
        age = df_v['Age']

        fig = go.Figure()

        # 5–95 band
        fig.add_trace(go.Scatter(
            x=pd.concat([age, age.iloc[::-1]]),
            y=pd.concat([df_v['centile_95'], df_v['centile_5'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(80,100,200,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='5th–95th centile',
        ))

        # 25–75 band
        fig.add_trace(go.Scatter(
            x=pd.concat([age, age.iloc[::-1]]),
            y=pd.concat([df_v['centile_75'], df_v['centile_25'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(80,100,200,0.25)',
            line=dict(color='rgba(0,0,0,0)'),
            name='25th–75th centile',
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=age,
            y=df_v['centile_50'],
            mode='lines',
            line=dict(color='rgb(60,80,180)', width=2),
            name='Median (50th)',
        ))

        fig.update_layout(
            xaxis_title='Age',
            yaxis_title=sel_var,
            height=500,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )

        st.plotly_chart(fig, use_container_width=True)
