from typing import Any
import time

import numpy as np
import streamlit as st
import os
import pandas as pd
import gui.utils_varnames as utilvars
import utils.utils_upload as utilup

from utils.utils_logger import setup_logger
logger = setup_logger()

##@st.cache_data  # type:ignore
def read_csv(fname):
    try:
        df = pd.read_csv(fname)
        return df
    except SystemExit as e:
        st.warning(f'Could not load data file: {os.path.basename(fname)}')
        return None
    
def load_centile_data(ptype):
    """
    Load centile data from the reference plots configuration and save in session state.
    """
    ss = st.session_state
    pdict = ss[ptype]

    cent_type = pdict['data']['cent_type']
    csv_cent = pdict['data']['cent_path'] + f"_{cent_type}.csv"

    if cent_type == 'None':
        pdict['data']['csv_cent'] = csv_cent
        pdict['data']['df_cent'] = None
        return 0

    else:
        # Do not reload if centile file is the same
        if csv_cent == pdict['data'].get('csv_cent', None):
            return 0

        df_cent = read_csv(csv_cent)
        col_dict = create_var_table(df_cent['VarName'].unique().tolist())

        try:
            df_cent = read_csv(csv_cent)
            col_dict = create_var_table(df_cent['VarName'].unique().tolist())
            pdict['data']['df_cent'] = df_cent
            pdict['data']['col_dict'] = col_dict    
            pdict['data']['csv_cent'] = csv_cent
            return 1

        except Exception as exc:
            st.toast(f'Could not load centile data: {exc}')
            st.toast(csv_cent)
            logger.debug(f'AAAA {csv_cent}')
            return 0

def _calc_normative_scores(df_data, df_cent):
    """
    For each variable in df_cent that exists in df_data, interpolate the
    centile curves at each subject's age and compute the percentile of their value.
    Returns a dataframe with MRID + one percentile column per variable.
    """
    centile_levels = [5, 25, 50, 75, 95]
    centile_cols = [f'centile_{p}' for p in centile_levels]

    vars_in_cent = df_cent['VarName'].unique()
    vars_to_map = [v for v in vars_in_cent if v in df_data.columns]

    new_cols = {}

    for var in vars_to_map:
        df_cv = df_cent[df_cent['VarName'] == var].sort_values('Age')
        if not all(c in df_cv.columns for c in centile_cols):
            continue
        ages = df_cv['Age'].values
        cent_curves = [df_cv[c].values for c in centile_cols]

        def get_pct(row, ages=ages, cent_curves=cent_curves):
            val = row[var]
            age = row['Age']
            if pd.isna(val) or pd.isna(age):
                return np.nan
            cent_at_age = [float(np.interp(age, ages, curve)) for curve in cent_curves]
            return float(np.interp(val, cent_at_age, centile_levels))

        new_cols[var] = df_data.apply(get_pct, axis=1)

    result = pd.concat([df_data[['MRID']], pd.DataFrame(new_cols, index=df_data.index)], axis=1)

    return result, vars_to_map

def _get_file_status(data_dir, entry):
    """Return (exists: bool, fpath: str) for a project file entry."""
    fpath = os.path.join(data_dir, entry['file'])
    return os.path.isfile(fpath), fpath

def _check_vars(df, expected_vars):
    """
    Check which expected variables are present/missing in df.
    Returns (present: list, missing: list).
    A lone '*' means any columns are acceptable.
    """

    if '*' in expected_vars:
        present = [v for v in df.columns]
        missing = [v for v in expected_vars if v != '*' and v not in df.columns]
    else:
        present = [v for v in expected_vars if v in df.columns]
        missing = [v for v in expected_vars if v not in df.columns]

    return present, missing

# ---------------------------------------------------------------------------
# Tab: View Data Files
# ---------------------------------------------------------------------------

def _panel_view_data_files():
    """
    Show all expected CSV files with their status (found / missing),
    expected variables, validation result, and optional content preview.
    """
    ss = st.session_state
    prj_files = ss.dicts.get('project_files', {})
    prj_vars  = ss.dicts.get('project_vars', {})
    data_dir  = ss.paths.get('curr_data')

    csv_entries = {k: v for k, v in prj_files.items() if v.get('type') == 'csv'}

    if not csv_entries:
        st.info('No CSV data files are defined for this project.')
        return

    for key, entry in csv_entries.items():
        exists, fpath = _get_file_status(data_dir, entry)
        desc = entry.get('desc', '')
        icon = ':material/check_circle:' if exists else ':material/cancel:'
        icon_color = 'green' if exists else 'red'

        with st.container(border=True):
            # Header row: icon · name · description
            c_icon, c_name, c_desc = st.columns([0.4, 1.6, 5])
            with c_icon:
                st.markdown(
                    f"<span style='color:{icon_color};font-size:1.3em'>{icon}</span>",
                    unsafe_allow_html=True,
                )
            with c_name:
                st.markdown(f"**{key}**")
            with c_desc:
                st.caption(desc)

            if exists:
                df = pd.read_csv(fpath)

                # Metadata + variable validation
                var_entry = prj_vars.get(key, {})
                expected_vars = var_entry.get('vars', [])

                c_meta, c_vars = st.columns([1, 3])
                with c_meta:
                    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

                if expected_vars:
                    _, missing = _check_vars(df, expected_vars)
                    with c_vars:
                        if missing:
                            with st.container(border=False, height=100):
                                st.warning(f"Missing: `{'`, `'.join(missing)}`")
                        else:
                            st.success('All expected variables present', icon=':material/done_all:')

                if st.toggle('Show contents', key=f'_tog_view_{key}'):
                    st.dataframe(df, use_container_width=True)

            else:
                st.caption(f':grey[Not found → `{entry["file"]}`]')


# ---------------------------------------------------------------------------
# Tab: Upload Data File
# ---------------------------------------------------------------------------

def _panel_upload_data_file():
    """
    Let the user pick one of the expected CSV files, upload a replacement,
    validate it contains all required variables, and save it to the project folder.
    """
    ss = st.session_state
    prj_files = ss.dicts.get('project_files', {})
    prj_vars  = ss.dicts.get('project_vars', {})
    data_dir  = ss.paths.get('curr_data')

    csv_entries = {k: v for k, v in prj_files.items() if v.get('type') == 'csv'}

    if not csv_entries:
        st.info('No CSV data files are defined for this project.')
        return

    sel_key = st.selectbox(
        'Select the file to upload / replace',
        list(csv_entries.keys()),
        key='_upload_sel_file',
    )

    if sel_key is None:
        return

    entry     = csv_entries[sel_key]
    var_entry = prj_vars.get(sel_key, {})
    expected  = var_entry.get('vars', [])
    fpath     = os.path.join(data_dir, entry['file'])

    with st.container(border=True):
        st.markdown(f"**{sel_key}** — {entry.get('desc', '')}")

        if expected and '*' not in expected:
            with st.container(border=False, height=100):
                st.caption('Expected variables: ' + ', '.join(f'`{v}`' for v in expected))

        if os.path.isfile(fpath):
            df_curr = pd.read_csv(fpath)
            st.caption(f"Current file: {len(df_curr):,} rows · {len(df_curr.columns)} columns")
        else:
            st.caption(':orange[No current file — slot is empty.]')

    uploaded = st.file_uploader(
        f'Upload CSV for **{sel_key}**',
        type=['csv'],
        key=f'_uploader_{sel_key}',
    )

    flag_check_vars = st.checkbox('Check uploaded file for expected variables', key='_chk_check_upload', value=True)

    if uploaded is None:
        return

    df_up = pd.read_csv(uploaded)

    with st.container(border=True):
        st.markdown('**Uploaded file**')
        st.caption(f"{len(df_up):,} rows · {len(df_up.columns)} columns")

        ok_to_save = True

        if  flag_check_vars:
            if expected and '*' not in expected:
                _, missing = _check_vars(df_up, expected)
                if missing:
                    with st.container(border=False, height=100):
                        st.error(f"Missing required columns: `{'`, `'.join(missing)}`")
                        ok_to_save = False
                else:
                    st.success('All required variables found.')

        if ok_to_save:
            if st.button('Save file to project folder', key='_btn_save_upload', type='primary'):
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                df_up.to_csv(fpath, index=False)
                st.success(f"Saved to `{entry['file']}`")
                # st.rerun()

        else:
            st.error('File not saved to project folder: Please fix the issues with the uploaded file!')

# ---------------------------------------------------------------------------
# Tab: Upload Data File
# ---------------------------------------------------------------------------

def _panel_config_dataprep():
    """
    Configuration options for data preparation steps, eg. ICV correction.
    """
    st.warning('Data preparation configuration coming soon...')

# ---------------------------------------------------------------------------
# Tab: Prepare Data
# ---------------------------------------------------------------------------

def corr_roi_icv(df, icv_col, mean_icv, roi_prefix='dlmuse-vol_', new_prefix='dlmuse-vol-icvcorr_'):
    """
    Correct each ROI volume column in df for intracranial volume (ICV).
    Returns a new dataframe with corrected columns.
    """
    list_cols = [c for c in df.columns if c.startswith(roi_prefix)]
    vol_cols = [c for c in list_cols if c != icv_col]

    df_corr = df.copy()
    for col in vol_cols:
        df_corr[col] = df[col] / df[icv_col] * mean_icv
    
    df_corr.columns = df_corr.columns.str.replace(roi_prefix, new_prefix)

    return df_corr

def create_var_table(cols):
    """
    Create a variable table from the dataframe columns, with labels and types.
    This is used for plotting and variable selection.
    """
    ss = st.session_state
    var_table = utilvars.DataVarColumns(cols)
    var_table.rename_roi_columns('muse', 'dlmuse-vol_', ss.dicts['muse']['map'])
    return var_table

def _run_data_preparation(available, data_dir):
    """
    Execute each preparation step inside an st.status block so the user
    can follow along.  Returns True on full success, False otherwise.
    """
    ss = st.session_state
    pdict = ss['user_plots']

    # step_sleep_short = 4  # Artificial delay for demonstration; remove in production
    # step_sleep_long = 8  # Artificial delay for demonstration; remove in production
    step_sleep_short = 0.5  # Artificial delay for demonstration; remove in production
    step_sleep_long = 2  # Artificial delay for demonstration; remove in production

    with st.status('Running data preparation…', expanded=True) as status_widget:

        # ---- Step 1: Load files ----
        st.write(':material/folder_open: Loading data files…')
        try:
            dfs = {}
            for key, entry in available.items():
                fpath = os.path.join(data_dir, entry['file'])
                dfs[key] = pd.read_csv(fpath)
            st.write(f'  → Loaded {len(dfs)} file(s).')
            time.sleep(step_sleep_short)

        except Exception as exc:
            status_widget.update(label='Load failed', state='error', expanded=True)
            st.error(f'Could not load files: {exc}')
            return False

        # ---- Step 2: ICV correction ----
        st.write(':material/calculate: Correcting ROI volumes for ICV')
        df_sel = dfs.get('dlmuse-vol', None)
        icv_col  = 'dlmuse-vol_702'
        mean_icv = ss.params.get('mean_icv', 1430000)

        df_corr = corr_roi_icv(df_sel, icv_col, mean_icv, 'dlmuse-vol_', 'dlmuse-vol-icvcorr_')
        dfs['dlmuse-vol-icvcorr'] = df_corr
        st.write(f'  → ICV correction applied to {len(df_corr.columns)} columns.')
        time.sleep(step_sleep_short)

        try:
            df_corr = corr_roi_icv(df_sel, icv_col, mean_icv, 'dlmuse-vol_', 'dlmuse-vol-icvcorr_')
            dfs['dlmuse-vol-icvcorr'] = df_corr
            st.write(f'  → ICV correction applied to {len(df_corr.columns)} columns.')
            time.sleep(step_sleep_short)

        except Exception as exc:
            st.warning(f'ICV correction failed (continuing): {exc}')

        # ---- Step 3: Merge files ----
        st.write(':material/merge: Merging files on MRID…')
        try:
            df_list = list(dfs.values())
            df = df_list[0]
            for d in df_list[1:]:
                df = df.merge(d, on='MRID', how='left')
            df['grouping_var'] = 'Data'
            st.write(f'  → Merged into {len(df):,} rows · {len(df.columns)} columns.')
            time.sleep(step_sleep_short)

        except Exception as exc:
            status_widget.update(label='Merge failed', state='error', expanded=True)
            st.error(f'Could not merge files: {exc}')
            return False

        # ---- Step 4: Centile calculation ----
        st.write(':material/calculate: Calculating normative scores based on centiles')
        df_sel = df

        try:
            # Calculate normative scores for all variables that have centile data
            df_cent = pdict['data'].get('df_cent', None)
            if df_cent is not None:
                df_pct, vars_mapped = _calc_normative_scores(df_sel, df_cent)
                # dfs['dlmuse-vol-pct'] = df_pct
                st.write(f'  → Normative scores calculated for {len(vars_mapped)} variables.')
            time.sleep(step_sleep_short)

        except Exception as exc:
            st.warning(f'Centile calculation failed (continuing): {exc}')

        if df_cent is None:
            st.warning('No centile data available for normative score calculation (continuing).')

        else:
            # ---- Step 5: Merge normative values ----
            st.write(':material/merge: Merging centile values...')
            try:
                df = df.merge(df_pct, on='MRID', how='left', suffixes=('', '_cent'))
                st.write(f'  → Merged into {len(df):,} rows · {len(df.columns)} columns.')
                time.sleep(step_sleep_short)

            except Exception as exc:
                status_widget.update(label='Merge failed', state='error', expanded=True)
                st.error(f'Could not merge files: {exc}')
                return False

        # ---- Step 6: Save plot data ----
        st.write(':material/merge: Saving plot data...')
        try:
            out_dir = os.path.join(data_dir, 'plot_data')
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, 'nichart_results.csv'), index=False)
            pdict['data']['df_user'] = df
            st.write(f'  → Plot data saved with {len(df):,} rows · {len(df.columns)} columns.')
            time.sleep(step_sleep_short)

        except Exception as exc:
            status_widget.update(label='Save failed', state='error', expanded=True)
            st.error(f'Could not save plot data: {exc}')
            return False

        # ---- Step 7: Build variable table ----
        st.write(':material/table_chart: Building variable table…')
        try:
            df = pdict['data'].get('df_user', None)
            pdict['data']['col_dict'] = create_var_table(df.columns.tolist())
            st.write(f'  → Variable table created')
            time.sleep(step_sleep_short)

        except Exception as exc:
            st.warning(f'Variable table step failed (continuing): {exc}')
            time.sleep(5)

        time.sleep(step_sleep_long)
        status_widget.update(label='Data preparation complete!', state='complete', expanded=True)

    return True

def _panel_prepare_data():
    """
    Multi-step data preparation tab.  Shows a summary of available files,
    a run button, and the current result when data is ready.
    """
    ss = st.session_state
    pdict = ss['user_plots']

    st.info(
        'Merge your project data files into a single dataset ready for plotting.', icon=':material/info:',
    )

    ss = st.session_state
    prj_files = ss.dicts.get('project_files', {})
    data_dir  = ss.paths.get('curr_data')

    csv_entries = {k: v for k, v in prj_files.items() if v.get('type') == 'csv'}
    available = {
        k: v for k, v in csv_entries.items()
        if os.path.isfile(os.path.join(data_dir, v['file']))
    }
    missing_files = {k: v for k, v in csv_entries.items() if k not in available}

    # # --- File summary ---
    # # with st.container(border=True):
    # #     st.markdown('**Data files**')
    # with st.expander('Data files', expanded=False):

    #     steps_info = [
    #         ('Load data files',             'Read each available CSV from the project folder.'),
    #         ('Correct ROI volumes for ICV', 'Normalise brain region volumes by intracranial volume.'),
    #         ('Merge files',                 'Join all files on the shared MRID key.'),
    #         ('Create variable table',       'Build a labelled variable mapping for plotting.'),
    #     ]

    #     if available:
    #         for k in available:
    #             desc = csv_entries[k].get('desc', '')
    #             st.caption(f':material/check_circle: **{k}** — {desc}')
    #     if missing_files:
    #         for k in missing_files:
    #             desc = csv_entries[k].get('desc', '')
    #             st.caption(f':material/cancel: :grey[**{k}** — {desc} *(not found)*]')

    # if not available:
    #     st.warning('No data files found in the project folder. Use the **Upload** tab to add files.')
    #     return

    # # --- Preparation steps summary ---
    # with st.expander('Preparation steps', expanded=False):
    #     for label, desc in steps_info:
    #         c1, c2 = st.columns([2, 4])
    #         c1.markdown(f'**{label}**')
    #         c2.caption(desc)

    # --- Run button ---
    if st.button('Prepare Data', key='_btn_run_dataprep'):
        success = _run_data_preparation(available, data_dir)
        if success:
            st.rerun()

    # --- Result ---
    if pdict['data'].get('df_user') is not None:
        df = pdict['data']['df_user']
        st.success(
            f'Data ready: {len(df):,} rows · {len(df.columns)} columns.',
            icon=':material/done_all:',
        )
        if st.toggle('Show prepared data', key='_tog_show_prep_data'):
            # with st.container(border=True, height=400):
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Public entry point (called from nichart_results.py)
# ---------------------------------------------------------------------------

def panel_prepare_data():
    """
    Prepare Data panel: view files, optionally upload replacements,
    and run multi-step data preparation.
    """
    logger.debug('    Function: panel_prepare_data')

    ss = st.session_state
    prj_data = ss.dicts.get('project_files', {})
    data_dir = ss.paths.get('curr_data')

    if not prj_data or not data_dir:
        st.warning('No project data dictionary or data directory configured.')
        return

    tab_prj, tab_run, tab_set = st.tabs(
        [":material/folder: Select Project",
         ":material/folder: Prepare Data for Charting",
         ":material/folder: Data & Settings"
        ],
        on_change='rerun',
    )

    if tab_prj.open:
        with tab_prj:
            utilup.panel_project_folder()

    if tab_run.open:
        with tab_run:
            _panel_prepare_data()

    if tab_set.open:
        with tab_set:
            tab_view, tab_upload, tab_config = st.tabs(
                ['View Files', 'Upload Files', 'Configure'],
                on_change='rerun',
            )
        if tab_view.open:
            with tab_view:
                _panel_view_data_files()

        if tab_upload.open:
            with tab_upload:
                _panel_upload_data_file()

        if tab_config.open:
            with tab_config:
                _panel_config_dataprep()
