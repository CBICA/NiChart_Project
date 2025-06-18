import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_io as utilio
import utils.utils_session as utilss
import os
import pandas as pd

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

def data_overview(in_dir, df_outdirs):

    st.markdown('##### Input Lists:')
    for dname in df_outdirs[df_outdirs.dtype == 'in_csv'].dname.tolist():
        dpath = os.path.join(
            in_dir, dname, dname + '.csv'
        )
        if os.path.exists(dpath):
            df = pd.read_csv(dpath)
            st.write(f'📁 `{dname + '.csv'}` — {df.shape[0]} rows × {df.shape[1]} columns')

    st.markdown('##### Input Images:')
    for dname in df_outdirs[df_outdirs.dtype == 'in_img'].dname.tolist():
        dpath = os.path.join(
            in_dir, dname
        )
        if os.path.exists(dpath):
            nimg = count_files_with_suffix(dpath, ['.nii.gz', '.nii'])
            st.write(f'📁 `{dname}` — {nimg} image files')
    
    st.markdown('##### Output Images:')
    for dname in df_outdirs[df_outdirs.dtype == 'out_img'].dname.tolist():
        dpath = os.path.join(
            in_dir, dname
        )
        if os.path.exists(dpath):
            nimg = count_files_with_suffix(dpath, ['.nii.gz', '.nii'])
            st.write(f'📁 `{dname}` — {nimg} image files')

    st.markdown('##### Output Lists:')
    for dname in df_outdirs[df_outdirs.dtype == 'out_csv'].dname.tolist():
        dpath = os.path.join(
            in_dir, dname, dname + '.csv'
        )
        if os.path.exists(dpath):
            df = pd.read_csv(dpath)
            st.write(f'📁 `{dname + '.csv'}` — {df.shape[0]} rows × {df.shape[1]} columns')

def data_merge(in_dir, df_outdirs, primary_key = 'MRID'):

    list_csv = []
    for dname in df_outdirs[df_outdirs.dtype.str.contains('csv')].dname.tolist():
        dpath = os.path.join(
            in_dir, dname, dname + '.csv'
        )
        if os.path.exists(dpath):
            list_csv.append(dname)
        
    if len(list_csv) == 0:
        st.warning('No data file to merge!')
        return

    sel_csv = st.pills(
        'Select csv data',
        list_csv,
        default = list_csv,
        selection_mode = 'multi',
        label_visibility = 'collapsed',
    )
    
    if sel_csv is None:
        return

    if st.button('Merge'):
        
        st.write(sel_csv)
        
        df_all = []
        for dname in sel_csv:
            try:
                dpath = os.path.join(
                    in_dir, dname, dname + '.csv'
                )
                df = pd.read_csv(dpath)
                df_all.append(df)
            except Exception as e:
                st.error(f"Failed to read {fname}: {e}")

        if df_all:
            # Merge on the primary key
            merged_df = df_all[0]
            for df in df_all[1:]:
                merged_df = pd.merge(merged_df, df, on=primary_key, how="outer")

            st.success(f"✅ Merged DataFrame has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
            st.dataframe(merged_df.head(10))

            return merged_df
