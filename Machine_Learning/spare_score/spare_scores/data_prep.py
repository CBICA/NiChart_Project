import logging
import os
import random
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from spare_scores.util import convert_to_number_if_possible


def check_train(df: pd.DataFrame, 
                                predictors: list,
                                to_predict: str,
                                key_var: str,
                                pos_group: str = '',
                                verbose: int = 1) -> Tuple[pd.DataFrame, list, str]:
    """Checks training dataframe for errors.

    Args:
        df: a pandas dataframe containing training data.
        predictors: a list of predictors for SPARE model training.
        to_predict: variable to predict.
        pos_group: group to assign a positive SPARE score (only for classification).

    Returns:
        a tuple containing 1) filtered dataframe, 2) filtered predictors, 3) SPARE model type.
    """
    # GAI 26/04/2023: Removed check for existence of these columns
    # if not {'ID','Age','Sex'}.issubset(set(df.columns)):
    #   logging.error('Please check required columns: ID, Age, Sex.')
    #   return 'Please check required columns: ID, Age, Sex.'
    if not set(predictors).issubset(df.columns):
        logging.error('Not all predictors exist in the input dataframe.')
        return 'Not all predictors exist in the input dataframe.'
    if to_predict not in df.columns:
        logging.error('Variable to predict is not in the input dataframe.')
        return 'Variable to predict is not in the input dataframe.'
    if to_predict in predictors:
        logging.info('Variable to predict is in the predictor set. This will be removed from the set.')
        predictors.remove(to_predict)
    if np.sum(np.sum(pd.isna(df[predictors]))) > 0:
        logging.info('Some participants have invalid predictor variables (i.e. n/a). They will be excluded.')
        df = df.loc[np.sum(pd.isna(df[predictors]), axis=1) == 0].reset_index(drop=True)

    if len(df[to_predict].unique()) == 2:
        if pos_group == '':
            logging.error('"pos_group" not provided (group to assign a positive score).')
            return '"pos_group" not provided (group to assign a positive score).'
        elif convert_to_number_if_possible(pos_group) not in df[to_predict].unique():
            logging.error('"pos_group" is not one of the two groups in the variable to predict.')
            return '"pos_group" is not one of the two groups in the variable to predict.'
        if np.min(df[to_predict].value_counts()) < 10:
            logging.error('At least one of the groups to classify is too small (n<10).')
            return 'At least one of the groups to classify is too small (n<10).'
        elif np.min(df[to_predict].value_counts()) < 100:
            logging.warn('At least one of the groups to classify may be too small (n<100).')
        mdl_task = 'Classification'

    elif len(df[to_predict].unique()) > 2:
        if df[to_predict].dtype not in ['int64', 'float64']:
            logging.error('Variable to predict must be either binary or numeric.')
            return 'Variable to predict must be either binary or numeric.'
        if len(df.index) < 10:
            logging.error('Sample size is too small (n<10).')
            return 'Sample size is too small (n<10).'
        elif len(df.index) < 100:
            logging.warn('Sample size may be too small (n<100).')
        if pos_group != '':
            logging.info('SPARE regression does not need a "pos_group". This will be ignored.')
        mdl_task = 'Regression'
    else:
        logging.error('Variable to predict has no variance.')
        return 'Variable to predict has no variance.'

    return df, predictors, mdl_task

def check_test(df: pd.DataFrame, 
                             meta_data: dict,
                             verbose: int = 1):
    """Checks testing dataframe for errors.

    Args:
        df: a pandas dataframe containing testing data.
        meta_data: a dictionary containing training information on its paired SPARE model.
    """
    ############# Removing the hardcoded check for the below cols #############
    # if not {'ID','Age','Sex'}.issubset(set(df.columns)):
    #   return logging.error('Please check required columns: ID, Age, Sex.')
    
    if not set(meta_data['predictors']).issubset(df.columns):
        cols_not_found = sorted(set(meta_data['predictors']) - set(df.columns))
        err = 'Not all predictors exist in the input dataframe: ' + str(cols_not_found)
        logging.error(err)
        return (err, cols_not_found)
    
    ############# Removing the hardcoded Age checks #############
    # if 'Age' not in df.columns:
    #   logging.info('"Age" column not found in the input dataframe.')
    # else:
    #   if (np.min(df['Age']) < np.min((meta_data['cv_results']['Age']))) or (
    #           np.max(df['Age']) > np.max((meta_data['cv_results']['Age']))):
    #     logging.warn('Some participants fall outside the age range of the SPARE model.')

    if np.sum(np.sum(pd.isna(df[meta_data['predictors']]))) > 0:
        logging.warn('Some participants have invalid (missing or NaN values) predictor variables.')

    ############# Removing the hardcoded ID checks #############
    if 'ID' not in df.columns:
        # logging.info('"ID" column not found in the input dataframe. Treating all participants as independent from training.')
        pass
    else:
        if np.any(df['ID'].isin(meta_data['cv_results']['ID'])):
            logging.info('Some participants seem to have been in the model training.')
    return 'OK', None

def smart_unique(df1: pd.DataFrame,
                                 df2: pd.DataFrame=None,
                                 to_predict: str=None,
                                 verbose: int=1) -> Union[pd.DataFrame, tuple]:
    """Select unique data points in a way that optimizes SPARE training.
    For SPARE regression, preserve data points with extreme values.
    For SPARE classification, preserve data points that help age match.

    Args:
        df1: a pandas dataframe.
        df2: a pandas dataframe (optional) if df1 and df2 are two groups to classify.
        to_predict: variable to predict. Binary for classification and continuous for regression.
            Must be one of the columnes in df. Ignored if df2 is given.

    Returns:
        a trimmed pandas dataframe or a tuple of two dataframes with only one time point per ID.
    """
    assert (isinstance(df2, pd.DataFrame) or (df2 is None)), (
        'Either provide a 2nd pandas dataframe for the 2nd argument or specify it with "to_predict"')
    if df2 is None:
        if to_predict is None:
            logging.error('Either provide a second dataframe or provide a column "to_predict"')
            return 'Either provide a second dataframe or provide a column "to_predict"'
        if len(df1[to_predict].unique()) < 2:
            logging.error('Variable to predict has no variance.')
            return 'Variable to predict has no variance.'
        if len(df1[to_predict].unique()) > 2:
            if ~np.any(df1['ID'].duplicated()):
                logging.info('No duplicated IDs.')
            else:
                logging.info('Select unique time points for SPARE regression training.')
                df1[f'{to_predict}_from_mean'] = np.abs(df1[to_predict] - np.mean(df1[to_predict]))
                df1 = (df1[df1.groupby('ID')[f'{to_predict}_from_mean']
                                            .transform(max) == df1[f'{to_predict}_from_mean']]
                                            .drop(columns=f'{to_predict}_from_mean'))
                df1 = df1[~df1['ID'].duplicated()].reset_index(drop=True)
            return df1
        if ~np.any(df1.groupby(['ID', to_predict]).size() > 1):
            logging.info('No duplicated IDs in either group.')
            return df1
        grps = list(df1[to_predict].unique())
        df1, df2 = df1[df1[to_predict] == grps[0]], df1[df1[to_predict] == grps[1]]
        no_df2 = True
    else:
        if to_predict is not None:
            logging.info('"to_predict" will be ignored.')
        if (~np.any(df1['ID'].duplicated())) and (~np.any(df2['ID'].duplicated())):
            logging.info('No duplicated IDs in either group.')
            return (df1, df2)
        no_df2 = False

    logging.info('Select unique time points for SPARE classification training.')
    swap = False
    if stats.ttest_ind(df1['Age'], df2['Age']).pvalue < 0.05:
        if np.mean(df1['Age']) < np.mean(df2['Age']):
                df1, df2, swap = df2.copy(), df1.copy(), True
        df2 = df2.loc[df2['Age'] >= np.min(df1['Age'])].reset_index(drop=True)
        df1 = df1[df1.groupby('ID')['Age'].transform(min) == df1['Age']].reset_index(drop=True)
        df2 = df2[df2.groupby('ID')['Age'].transform(max) == df2['Age']].reset_index(drop=True)
    else:
        logging.info('Age difference not significant between two groups.')
    df1 = df1[~df1['ID'].duplicated()].reset_index(drop=True)
    df2 = df2[~df2['ID'].duplicated()].reset_index(drop=True)
    if swap:
        df1, df2 = df2.copy(), df1.copy()
    return pd.concat([df1, df2], ignore_index=True) if no_df2 else (df1, df2)

def age_sex_match(df1: pd.DataFrame,
                                    df2: pd.DataFrame = None,
                                    to_match: str = None,
                                    p_threshold: float = 0.15,
                                    verbose: int = 1,
                                    age_out_percentage: float = 20) -> pd.DataFrame:
    """Match two groups for age and sex.

    Args:
        df1: a pandas dataframe.
        df2: a pandas dataframe (optional) if df1 and df2 are two groups to classify.
        to_match: a binary variable of two groups. Must be one of the columns in df.
            Ignored if df2 is given.
            If to_match is 'Sex', then only perform age matching.
        p_threshold: minimum p-value for matching.
        verbose: whether to output messages.
        age_out_percentage: percentage of the larger group to randomly select a participant to
            take out from during the age matching. For example, if age_out_percentage = 20 and the
            larger group is significantly older, then exclude one random participant from the fifth
            quintile based on age.

    Returns:
        a trimmed pandas dataframe or a tuple of two dataframes with age/sex matched groups.
    """
    assert (isinstance(df2, pd.DataFrame) or (df2 is None)), (
        'Either provide a 2nd pandas dataframe for the 2nd argument or specify the two groups with "to_match"')

    if df2 is None:
        if to_match is None:
            logging.error('Either provide a 2nd dataframe or provide a column "to_match"')
            return 'Either provide a 2nd dataframe or provide a column "to_match"'
        if len(df1[to_match].unique()) != 2:
            logging.error('Variable to match must be binary')
            return 'Variable to match must be binary'
        grps = list(df1[to_match].unique())
        df1, df2 = df1[df1[to_match] == grps[0]], df1[df1[to_match] == grps[1]]
        no_df2 = True
    else:
        if to_match is not None:
            logging.info('Two dataframes provided. "to_match" will be ignored.')
        no_df2 = False

    if (age_out_percentage <= 0) or (age_out_percentage >= 100):
        logging.error('Age-out-percentage must be between 0 and 100')
        return 'Age-out-percentage must be between 0 and 100'
    if (len(df1['Sex'].unique())==1) & (len(df2['Sex'].unique())==1):
        logging.info('Performing age matching only.')
        sex_match = False
    else:
        sex_match = True

    swap = 1
    random.seed(2022)
    n_orig = len(df1.index) + len(df2.index)

    p_age = stats.ttest_ind(df1['Age'], df2['Age']).pvalue
    if sex_match:
        s1, s2 = df1['Sex'].unique() 
        p_sex = stats.chi2_contingency([np.array(df1['Sex'].value_counts()), np.array(df2['Sex'].value_counts())])[1]
    else:
        p_sex = 1
    logging.debug(f' Original: P_age: {np.round(p_age,2)}/ P_sex: {np.round(p_sex,2)}')

    p_age_all, p_sex_all = np.array(p_age), np.array(p_sex)
    while np.min([p_age, p_sex]) < p_threshold:
        if len(df2.index) > len(df1.index):
            df1, df2 = df2.copy(), df1.copy()
            swap *= -1
        if p_age < p_threshold:
            if np.mean(df1['Age']) < np.mean(df2['Age']):
                i_age = df1['Age'] < np.percentile(df1['Age'], age_out_percentage)
            else:
                i_age = df1['Age'] > np.percentile(df1['Age'], 100-age_out_percentage)
        else:
            i_age = df1['Age'] >= 0
        if p_sex < p_threshold:
            if np.sum(df1['Sex'] == s1)/np.sum(df1['Sex'] == s2) > np.sum(df2['Sex'] == s1)/np.sum(df2['Sex'] == s2):
                i_sex = df1['Sex'] == s1
            else:
                i_sex = df1['Sex'] == s2
        else:
            i_sex = np.ones(len(df1.index)).astype(bool)

        try:
            df1 = df1.drop(random.sample(list(df1[i_age & i_sex].index), 1)).reset_index(drop=True)
        except:
            suggestion = 'Try increasing "age_out_percentage" parameter.' if np.min([len(df1.index), len(df2.index)]) > 10 else ''
            logging.error(f'Matching failed... {suggestion}')
            return f'Matching failed... {suggestion}'
        p_age = stats.ttest_ind(df1['Age'], df2['Age']).pvalue
        p_sex = stats.chi2_contingency([np.array(df1['Sex'].value_counts()), np.array(df2['Sex'].value_counts())])[1]
        p_age_all = np.append(p_age_all, p_age)
        p_sex_all = np.append(p_sex_all, p_sex)
    if swap == -1:
        df1, df2 = df2.copy(), df1.copy()

    logging.debug(f' {n_orig - len(df1.index) - len(df2.index)} participants excluded')
    logging.debug(f' Final: P_age: {np.round(p_age,2)}/ P_sex {np.round(p_sex,2)}')
    logging.info('Age/Sex matched!')
    if no_df2:
        return pd.concat([df1, df2], ignore_index=True)
    else:
        return (df1, df2)

def logging_basic_config(verbose=1, content_only=False, filename=''):
    logging_level = {0:logging.WARNING, 1:logging.INFO, 2:logging.DEBUG, 3:logging.ERROR, 4:logging.CRITICAL}
    fmt = ' %(message)s' if content_only else '%(levelname)s (%(funcName)s): %(message)s'
    if filename != '' and filename is not None:
        if not os.path.exists(filename):
            dirname, _ = os.path.split(filename)
            if dirname != '':
                os.mkdir(dirname)
        logging.basicConfig(level=logging_level[verbose], format=fmt, force=True, filename=filename)
    else:
        logging.basicConfig(level=logging_level[verbose], format=fmt, force=True)
    return logging.getLogger()
        
def convert_cat_variables(df, predictors, meta_data):
    cat_vars = [var 
                for var in df[predictors].columns 
                if df[var].dtypes == 'O']
    meta_data.categorical_var_map = {var: None for var in cat_vars}
    for var in cat_vars:
        if len(df[var].unique()) <= 2:
            meta_data.categorical_var_map[var] = {df[var].unique()[0]:  1, 
                                                  df[var].unique()[-1]: 2}
            df[var] = df[var].map(meta_data.categorical_var_map[var])
        
        elif len(df[var].unique()) > 2:
            raise ValueError('Categorical variables with more than 2 '
                             + 'categories are currently not supported.')
    return df, meta_data