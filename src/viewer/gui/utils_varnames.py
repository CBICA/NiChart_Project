from typing import Any

import pandas as pd
import streamlit as st

class VarMapper:
    """
    Utility class for bidirectional mapping between variable indices and names.

    Can be initialized with:
    - dict: {index: name}
    - pandas DataFrame with two columns

    This class provides:
    - Fast lookup from index → name and name → index
    - Support for single values and lists
    - Optional suffix/prefix handling when converting to names
    - Safe lookups with defaults for missing keys

    Example:
        mapping = {"pl1": "age", "pl2": "cholesterol"}
        mapper = VarMapper(mapping)

        mapper.to_name("pl1")              -> "age"
        mapper.to_index("age")             -> "pl1"
        mapper.to_names(["pl1"], "_x")     -> ["age_x"]
    """

    def __init__(self, data):
        # -------- Handle dict input --------
        if isinstance(data, dict):
            self.idx_to_name = dict(data)

        # -------- Handle DataFrame input --------
        elif isinstance(data, pd.DataFrame):
            data = data.astype(str)
            try:
                index_col = data.columns[0]
                name_col = data.columns[1]
            except IndexError:
                raise ValueError("DataFrame must have at least two columns")
    
            # Build mapping from DataFrame columns
            self.idx_to_name = dict(zip(data[index_col], data[name_col]))

        else:
            raise TypeError("Data must be dict or pandas DataFrame")

        # Reverse mapping
        self.name_to_idx = {v: k for k, v in self.idx_to_name.items()}

    # -------- Single conversions --------
    def to_name(self, idx, prefix='', suffix='', default=None):
        val = self.idx_to_name.get(idx, default)
        if val is None:
            return default
        return f"{prefix}{val}{suffix}"

    def to_index(self, name, default=None):
        return self.name_to_idx.get(name, default)

    # -------- List conversions --------
    def to_names(self, idx_list, prefix='', suffix='', default=None):
        return [
            self.to_name(i, prefix=prefix, suffix=suffix, default=default)
            for i in idx_list
        ]

    def to_indices(self, name_list, default=None):
        return [
            self.to_index(n, default=default)
            for n in name_list
        ]

    # -------- Optional helpers --------
    def has_index(self, idx):
        return idx in self.idx_to_name

    def has_name(self, name):
        return name in self.name_to_idx

class DataVarColumns:
    """
    Manages column names for a user data CSV DataFrame, providing
    renamed/display versions of columns for use in plotting.

    Attributes:
        df_cols (pd.DataFrame): Single-column DataFrame indexed by original
                                  column names, with a 'renamed' column holding
                                  display names. Initialised to the identity
                                  (renamed == index).
        dict_fwd (dict):            Forward mapping from original column names to display names.
        dict_rev (dict):            Reverse mapping from display names to original column names.
    """
    def df_to_dict(self):
        self.dict_fwd = self.df_cols['renamed'].to_dict()
        self.dict_rev = {v: k for k, v in self.dict_fwd.items()}

    def __init__(self, cols: list = None):
        self.df_cols = pd.DataFrame(
            {'renamed': cols, 'atlas': None, 'roi_index': None},
            index=pd.Index(cols, name='column')
        )
        self.df_to_dict()

    def rename_roi_columns(self, atlas, var_prefix, var_mapper):
        '''
        Rename columns in df_cols (based on variable type, eg. roi index to name).
        This is used to create user-friendly display names for plotting variables.
        '''
        df_cols = self.df_cols
        df_sel = df_cols[df_cols.index.str.startswith(var_prefix)]
        in_list = df_sel.index.str.replace(var_prefix, '').to_list()
        renamed = var_mapper.to_names(in_list, var_prefix)
        # st.dataframe(df_sel)
        df_sel['renamed'] = renamed
        df_sel['atlas'] = atlas
        df_sel['roi_index'] = in_list
        df_sel = df_sel.dropna()
        df_cols.update(df_sel)
        self.df_cols = df_cols
        self.df_to_dict()

    def column_to_roi_index(self, column_name):
        '''
        Get roi index for a given column name, if it corresponds to an roi variable.
        '''
        try:
            return int(self.df_cols.loc[column_name, 'roi_index'])
        except KeyError:
            return None

    def renamed_to_roi_index(self, column_name):
            '''
            Get roi index for a given renamed column name, if it corresponds to an roi variable.
            '''
            try:
                dft = self.df_cols.copy()
                dft = self.df_cols[self.df_cols['renamed']==column_name]
                return int(dft['roi_index'].tolist()[0])
            except KeyError:
                return None
