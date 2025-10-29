import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import os
import pandas as pd
import streamlit_antd_components as sac

from utils.utils_logger import setup_logger
logger = setup_logger()

def count_files_with_suffix(in_dir, suffixes):
    count = 0
    if isinstance(suffixes, list):
        suffixes = tuple(suffixes)    
    for file_name in os.listdir(in_dir):
        full_path = os.path.join(in_dir, file_name)
        if os.path.isfile(full_path) and file_name.endswith(suffixes):
            count += 1
    return count

def build_folder_tree(
    path,
    list_dirs = None,
    list_suff = None,
    file_limit=5,
    list_ignore = None,
    flag_dir_disabled = False,
):
    tree_items = []
    list_paths = []
    try:
        # Read files
        entries = os.listdir(path)

        # Sort using given list
        if list_dirs is not None:
            e1 = [x for x in st.session_state.out_dirs if x in entries]
            e2 = [x for x in entries if x not in st.session_state.out_dirs]
            entries = e1 + e2

        # Remove given items
        if list_ignore is not None:
            entries = [x for x in entries if x not in list_ignore]

        files = []
        dirs = []
        for name in entries:
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                # Only add if directory contains at least one file with given suffix
                list_files = os.listdir(full_path)
                if not list_suff or any(f.endswith(s) for f in list_files for s in list_suff):
                    dirs.append(name)
                
            else:
                if not list_suff or any(name.endswith(s) for s in list_suff):
                    files.append(name)

        # Add subfolders
        for d in dirs:
            full_path = os.path.join(path, d)            
            children, tmp_list = build_folder_tree(
                full_path, list_dirs, list_suff, file_limit, list_ignore, flag_dir_disabled
            )
            tree_items.append(
                sac.TreeItem(
                    label=d,
                    icon='folder',
                    children=children,
                    disabled=flag_dir_disabled,
                    tooltip=f'{len(children)} items inside'
                )
            )
            list_paths.append(full_path)
            list_paths = list_paths + tmp_list

        # Add files (limit shown files)
        for i, name in enumerate(files[:file_limit]):
            full_path = os.path.join(path, name)
            ext = os.path.splitext(name)[1].lower()
            tags = []
            if ext == '.csv':
                tags.append(sac.Tag('CSV', color='red'))

            tree_items.append(
                sac.TreeItem(
                    label=name,
                    icon='table',
                    #description=f'{ext[1:]} file',
                    tag=tags,
                    tooltip=f'File: {name}'
                )
            )
            list_paths.append(full_path)

        # Add collapsed summary node if too many files
        if len(files) > file_limit:
            tree_items.append(
                sac.TreeItem(
                    label=f"... and {len(files) - file_limit} more files",
                    icon='ellipsis',
                    disabled=True
                )
            )
            list_paths.append('...extra...')

    except Exception as e:
        st.error(f"Error reading {path}: {e}")

    return tree_items, list_paths

def data_overview(in_dir):
    '''
    Show files in data folder
    '''
    dname = os.path.basename(in_dir)
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.markdown("##### View Project Folder:")

    with col2:
        sac.buttons(
            [sac.ButtonsItem(label='Switch Folder')],
            label='', align='right', color='cyan'
        )

    #colb1, colb2 = st.columns([1,8])
    
    
    #with colb2:
    st.markdown(f"##### 📂 `{dname}`")

    if os.path.exists(in_dir):
        tree_items, list_paths = build_folder_tree(in_dir, st.session_state.out_dirs)
        selected = sac.tree(
            items=tree_items,
            #label='Project Folder',
            index=None,
            align='left', size='xl', icon='table',
            checkbox=False,
            #checkbox_strict = True,
            open_all = False,
            return_index = True
            #height=400
        )
        
        if selected:
            if isinstance(selected, list):
                selected = selected[0]
            fname = list_paths[selected]
            if fname.endswith('.csv'):
                try:
                    df_tmp = pd.read_csv(fname)
                    st.info(f'Data file: {fname}')
                    st.dataframe(df_tmp)
                except:
                    st.warning(f'Could not read csv file: {fname}')

    else:
        st.error(f"Folder `{in_dir}` not found.")

def select_files(in_dir):
    '''
    Merge data csv files
    '''
    if not os.path.exists(in_dir):
        return

    st.markdown(f"##### 📂 `{in_dir}`")
    tree_items, list_paths = build_folder_tree(
        in_dir,
        st.session_state.out_dirs,
        ['.csv'],
        5,
        ['data_merged'],
        True
    )
    selected = sac.tree(
        items=tree_items,
        #label='Project Folder',
        index=None,
        align='left', size='xl', icon='table',
        checkbox=True,
        checkbox_strict = True,
        open_all = True,
        return_index = True
        #height=400
    )

    list_csv = [list_paths[i] for i in selected]

    if st.button('Merge'):
        if len(list_csv) == 0:
            st.warning('No files to merge!')
            return

        else:
            print(list_csv)

        df_all = []
        for dname in list_csv:
            try:
                df = pd.read_csv(dname)
                
                ## FIXME: Custom editing in column names
                df.columns = df.columns.str.replace('DL_MUSE_Volume_','')
                df.columns = df.columns.str.replace('SurrealGAN_MRID','MRID')
                df.columns = df.columns.str.replace('SurrealGAN_','')
                df.columns = df.columns.str.replace('CCL-NMF','CCLNMF_')
                df.columns = df.columns.str.replace('SPARE_RG','SPARE_BA')
                df.columns = df.columns.str.replace('SPARE_CL_decision_function','SPARE_AD')
                df.columns = df.columns.str.replace('Prediction','DL_BrainAge')

            except Exception as e:
                st.error(f"Failed to read {dname}: {e}")

            # Rename columns if dict for data exists
            if dname.endswith('DLMUSE_Volumes.csv'):
                df = df.rename(
                    columns = st.session_state.dicts['muse']['ind_to_name']
                )
            df_all.append(df)

        if df_all:
            # Merge on the primary key
            primary_key = 'MRID'
            merged_df = df_all[0]
            for df in df_all[1:]:
                merged_df = pd.merge(
                    merged_df,
                    df,
                    on=primary_key,
                    how="outer",
                    suffixes = ['', '_tmpduplicate']
                )
                sel_cols = merged_df.columns[merged_df.columns.str.contains('_tmpduplicate')==False]
                merged_df = merged_df[sel_cols]

            st.success(f"✅ Merged DataFrame has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")

            # Save merged data
            out_dir = os.path.join(
                st.session_state.paths['project'], 'data_merged'
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            out_csv = os.path.join(
                out_dir, 'data_merged.csv'
            )
            
            try:
                merged_df.to_csv(out_csv)
                
                # Reset plot data
                utilss.init_plot_vars()
                
                
            except:
                st.error(f'Could not write merged data: {out_csv}')

