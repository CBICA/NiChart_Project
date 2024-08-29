import logging
import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import numpy.linalg as la
import copy
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
import pickle
#import dill
import logging
from typing import Union

## Set logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)      ## While debugging
#logger.setLevel(logging.INFO)    ## FIXME Comments will be removed in release version

## FIXME tmp
def save_to_pickle(fname, obj):
    out_file = open(fname, 'wb')
    pickle.dump(obj, out_file)
    out_file.close()

#####################################################################################
## Functions common to nh_learn_model and nh_apply_model

def read_data(in_data : Union[pd.DataFrame, str]):
    """ 
    Read initial data
    """
    ## Verify data and read to dataframe
    if isinstance(in_data, pd.DataFrame):
        df_in = in_data.copy()
    else:
        if os.path.exists(in_data) == False:
            logger.warning("File not found: " + in_data)
            return None
        else:
            try:
                df_in = pd.read_csv(in_data)
            except:
                logger.warning("Could not read data file: " + in_data)
                return None

    ## Reset index for data
    df_in = df_in.reset_index(drop = True)
    
    return df_in

def read_model(model : Union[dict, str]):
    """ 
    Read initial data
    """    
    ## Read model file
    if isinstance(model, str):
        try:
            fmdl = open(model, 'rb')
            out_model = pickle.load(fmdl)
            fmdl.close()
        except:
            logger.warning("Could not read model file: " + model)
            return None
    else:
        out_model = copy.deepcopy(model)
    
    return out_model

def verify_data_to_model(mdl_in, df_in):
    """ 
    Verify that model variables are included in the data
    """    
    dict_vars = mdl_in['mdl_ref']['dict_vars']
    mdl_vars = dict_vars['cov_columns'] + dict_vars['data_vars']
    missing_vars = [x for x in mdl_vars if x not in df_in.columns]
    
    return missing_vars


def filter_data(df_in, dict_cat):
    '''
        Filter data
    '''
    ## Remove rows with categorical values not present in model data
    num_sample_init = df_in.shape[0]
    for tmp_var in dict_cat.keys():   ## For each cat var from the dict_vars that was saved in model
        tmp_vals = dict_cat[tmp_var]         ##   Read values of the cat var
        df_in = df_in[df_in[tmp_var].isin(tmp_vals)]  ##   Remove rows with values different than those in model
    num_sample_new = df_in.shape[0]
    num_diff = num_sample_init - num_sample_new
    if num_diff != 0:
        logger.info('WARNING: Samples with categorical values not in model data are discarded ' + 
                    ', n removed = ' + str(num_diff))

    return df_in

def check_key(df_in, key_var):
    '''
    Check the primary key column
    '''
    ## Check if key column exists in data
    if key_var is not None: 
        if key_var not in df_in.columns:
            
            logger.info(df_in.columns)
            input()
            
            logger.error("Primary key not in data columns: " + key_var)
            return None

    ## If key is not entered as an input, first column is considered as key
    if key_var is None:
        key_var = df_in.columns.tolist()[0]
        logger.info("Primary key is set to: " + str(key_var))
    
    ## Check that the key column has unique values
    if df_in[key_var].unique().shape[0] != df_in.shape[0]:
        logger.error("Values for primary key are not unique: " + key_var)
        return None

    ## Return key var
    return key_var

def make_dict_vars(df_in, key_var, batch_var, num_vars, cat_vars, spline_vars, ignore_vars, data_vars):
    """ 
    Make a dictionary of all variables
    """
    ## Get variable lists
    all_columns = df_in.columns.tolist()
    cov_columns = [batch_var] + num_vars + cat_vars + spline_vars
    non_data_columns = [key_var] + cov_columns + ignore_vars
    if len(data_vars) == 0:
        data_vars = [x for x in all_columns if x not in non_data_columns]
    
    ## Create dictionary of covars    
    dict_vars = {'cov_columns' : cov_columns, 
                 'key_var' : key_var,
                 'batch_var' : batch_var,
                 'num_vars' : num_vars,
                 'cat_vars' : cat_vars,
                 'spline_vars' : spline_vars,
                 'ignore_vars': ignore_vars,
                 'data_vars': data_vars
                 }
    
    ## Return dictionary
    return dict_vars

def make_dict_batches(df_cov, batch_var):
    '''
        Create a dictionary with meta data about batches  
    '''    
    df_tmp = pd.get_dummies(df_cov[batch_var], prefix = batch_var)
    design_batch_indices = {}
    for bname in df_tmp.columns:
        design_batch_indices[bname] = df_tmp[df_tmp[bname] == 1].index.to_list()
    dict_batches = {'batch_values': df_cov[batch_var].unique().tolist(),
                    'design_batch_vars': df_tmp.columns.to_list(),
                    'n_batches': df_tmp.shape[1],
                    'n_samples': df_tmp.shape[0],
                    'n_samples_per_batch': np.array(df_tmp.sum(axis=0)),
                    'design_batch_indices': design_batch_indices}
    return dict_batches

def make_dict_cat(df_in, cat_vars):
    ## Make a dictionary of categorical variables
    
    ## Find unique values for each categorical variable
    dict_cat ={}
    for tmp_var in cat_vars:
        cat_vals = df_in[tmp_var].unique().tolist()
        dict_cat[tmp_var] = cat_vals 

    ## Return dictionary
    return dict_cat

def add_spline_bounds(df_in, spline_vars, dict_vars):
    """ 
    Calculate bounds for spline columns
    """
    dict_out = dict_vars.copy()
    
    ## Calculate spline bounds (min and max of each spline var)
    spline_bounds_min = []
    spline_bounds_max = []
    for tmp_var in spline_vars:
        spline_bounds_min.append(df_in[tmp_var].min())
        spline_bounds_max.append(df_in[tmp_var].max())
        
    ## Add spline bounds to dictionary
    dict_out['spline_bounds_min'] = spline_bounds_min
    dict_out['spline_bounds_max'] = spline_bounds_max
    
    ## Return dictionary
    return dict_out

def get_data_and_covars(df_in, dict_vars):
    '''
    Split dataframe into data and covars
    '''
    ## Extract covars and data dataframes
    try:
        df_cov = df_in[ [dict_vars['key_var']] + dict_vars['cov_columns']]
    except:
        logger.error("Could not extract covariate columns from input data: " + cov_columns)
        return None

    try:
        df_data = df_in[dict_vars['data_vars']]
    except:
        logger.error("Could not extract data columns from input data: " + dict_vars['data_vars'])
        return None

    ## Replace special characters in batch values
    ## FIXME Special characters fail in GAM formula
    ##   Update this using patsy quoting in next version
    batch_var = dict_vars['batch_var']
    df_cov.loc[:, batch_var] = df_cov[batch_var].str.replace('-', '_').str.replace(' ', '_')
    df_cov.loc[:, batch_var] = df_cov[batch_var].str.replace('.', '_').str.replace('/', '_')
    
    ## FIXME Remove null values in data dataframe (TODO)
    
    ## FIXME Remove null values in covar dataframe (TODO)
    
    return df_cov, df_data

def make_design_dataframe(df_cov, dict_vars):
    '''
    Expand the covariates dataframe adding columns that will constitute the design matrix
    New columns in the output dataframe are: 
        - one-hot matrix of batch variables (full)
        - one-hot matrix for each categorical var (removing the first column)
        - column for each continuous_vars
        - spline variables are skipped (added later in a separate function)
    '''
    ## Make output dataframe
    df_design_out = df_cov.copy()
    
    ## Make output dict
    dict_design = {}
    
    ## Keep columns that will be included in the final design matrix
    design_vars = []

    ## Add one-hot encoding of batch variable
    df_tmp = pd.get_dummies(df_design_out[dict_vars['batch_var']], prefix = dict_vars['batch_var'], dtype = float)
    design_batch_vars = df_tmp.columns.tolist()
    df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
    design_vars = design_vars + design_batch_vars

    ## Add numeric variables
    ##   Numeric variables do not need any manipulation; just add them to the list of design variables
    num_vars = dict_vars['num_vars']
    design_vars = design_vars + num_vars
    dict_design['num_vars'] = num_vars

    ## Add one-hot encoding for each categoric variable
    cat_vars = []
    for tmp_var in dict_vars['cat_vars']:
        df_tmp = pd.get_dummies(df_design_out[tmp_var], prefix = tmp_var, drop_first = True, 
                                dtype = float)
        df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
        cat_vars = cat_vars + df_tmp.columns.tolist()
    design_vars = design_vars + cat_vars
    dict_design['cat_vars'] = cat_vars
    
    ## Add dict item for non-batch columns
    dict_design['non_batch_vars'] = num_vars + cat_vars

    ## Return output vars
    return df_design_out[design_vars], dict_design

def adjust_data_final(df_s_data, df_gamma_star, df_delta_star, df_stand_mean, df_pooled_stats, dict_batches):
    '''
        Apply estimated harmonization parameters
    '''
    ## Create output df
    df_h_data = df_s_data.copy()
    
    ## For each batch
    for i, b_tmp in enumerate(dict_batches['design_batch_vars']):
        
        ## Get sample index for the batch
        batch_idxs = dict_batches['design_batch_indices'][b_tmp]

        ## Get batch updated
        df_denom = np.sqrt(df_delta_star.loc[b_tmp,:].astype(np.float64))
        df_numer = df_s_data.loc[batch_idxs] - df_gamma_star.loc[b_tmp]
        df_h_data.loc[batch_idxs] = df_numer.div(df_denom)

    ## Get data updated
    df_vsq = np.sqrt(df_pooled_stats.loc['var_pooled'].astype(np.float64))
    df_h_data = df_h_data.multiply(df_vsq) + df_stand_mean

    return df_h_data

def fit_LS_model(df_s_data, df_design, dict_batches, skip_emp_bayes = False):
    """
        Dataframe implementation of neuroCombat function fit_LS_model_and_find_priors
    """
    ## Get design matrix columns only for batch variables (batch columns)
    batch_vars = dict_batches['design_batch_vars']
    df_design_batch_only = df_design[batch_vars]
    
    ## Calculate gamma_hat
    df_tmp = df_design_batch_only.T.dot(df_design_batch_only)
    df_tmp.loc[:,:] = np.linalg.inv(np.matrix(df_tmp))    
    df_tmp = df_tmp.dot(df_design_batch_only.T)
    df_gamma_hat = df_tmp.dot(df_s_data)

    ## Calculate delta_hat
    df_delta_hat = pd.DataFrame(columns = df_gamma_hat.columns, index = df_gamma_hat.index)
    for i, b_tmp in enumerate(dict_batches['design_batch_vars']):
        batch_idxs = dict_batches['design_batch_indices'][b_tmp]
        df_delta_hat.loc[b_tmp, :] = df_s_data.loc[batch_idxs, :].var(axis = 0, ddof = 1)

    ## Calculate other params
    gamma_bar = None
    t2 = None
    a_prior = None
    b_prior = None
    if skip_emp_bayes == False:
        df_gamma_bar = df_gamma_hat.mean(axis = 1) 
        df_t2 = df_gamma_hat.var(axis=1, ddof=1)
        df_a_prior = calc_aprior(df_delta_hat)
        df_b_prior = calc_bprior(df_delta_hat)

    ## Create out dict
    dict_LS = {'df_gamma_hat' : df_gamma_hat, 'df_delta_hat' : df_delta_hat, 
               'df_gamma_bar' : df_gamma_bar, 'df_t2' : df_t2, 
               'df_a_prior' : df_a_prior, 'df_b_prior' : df_b_prior}

    return dict_LS

def calc_aprior(df):
    m = df.mean(axis = 1)
    s2 = df.var(axis = 1, ddof = 1)
    df_out = (2 * s2 +m**2) / s2 
    return df_out 

def calc_bprior(df):
    m = df.mean(axis = 1)
    s2 = df.var(axis = 1, ddof = 1)
    df_out = ( m * s2 + m * m * m ) / s2
    return df_out 

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    '''
    Iterative solver
    '''
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

def find_parametric_adjustments(df_s_data, dict_LS, dict_batches, skip_emp_bayes = False):
    '''
        Calculate adjusted gamma and delta values (gamma and delta star)
    '''
    ## Get data into numpy matrices
    s_data = np.array(df_s_data).T    

    if skip_emp_bayes:
        df_gamma_star = dict_LS['df_gamma_hat']
        delta_star = dict_LS['df_delta_hat']
    else:
        design_batch_indices = dict_batches['design_batch_indices'] 

        gamma_star, delta_star = [], []
        for i, b_tmp in enumerate(dict_batches['design_batch_vars']):
            
            ## Get data
            batch_idxs = dict_batches['design_batch_indices'][b_tmp]
            s_data = np.array(df_s_data.loc[batch_idxs, :]).T
            gamma_hat = np.array(dict_LS['df_gamma_hat'].loc[b_tmp, :])
            delta_hat = np.array(dict_LS['df_delta_hat'].loc[b_tmp, :])
            gamma_bar = np.array(dict_LS['df_gamma_bar'].loc[b_tmp])
            t2 = np.array(dict_LS['df_t2'].loc[b_tmp])
            a_prior = np.array(dict_LS['df_a_prior'].loc[b_tmp])
            b_prior = np.array(dict_LS['df_b_prior'].loc[b_tmp])
            
            ## Calculate parameters
            temp = it_sol(s_data, gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior)
            
            
            gamma_star.append(temp[0])
            delta_star.append(temp[1])
                        
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)
        
    df_gamma_star = pd.DataFrame(data = gamma_star, index = dict_batches['design_batch_vars'], 
                                 columns = df_s_data.columns.tolist())
    df_delta_star = pd.DataFrame(data = delta_star, index = dict_batches['design_batch_vars'], 
                                 columns = df_s_data.columns.tolist())

    return df_gamma_star, df_delta_star

def save_model(out_model, out_file_name):
    """
    Save model as a pickle file
    """
    out_file_name_full = os.path.abspath(out_file_name)
    out_file = open(out_file_name_full, 'wb')
    pickle.dump(out_model, out_file)
    out_file.close()

def save_data(out_df, out_file_name):
    """
    Save dataframe as a csv file
    """
    out_file_name_full = os.path.abspath(out_file_name)

    ## Save out file
    out_df.to_csv(out_file_name_full, index = False)


def calc_spline_model(df_spline, spline_bounds_min = None, spline_bounds_max = None, param_spline_doff = 10, param_spline_degree = 3):
    '''
    calculate spline model for the selected spline variables
    '''
    doff = [param_spline_doff] * df_spline.shape[1] 
    degree = [param_spline_degree] * df_spline.shape[1] 

    ## If spline_bounds are provided construct the knot_kwds arg
    ## knot_kwds arg for multiple variables is a "list of dictionaries" with min/max bound values
    if len(spline_bounds_min) > 0:        
        kwds = []
        for i, tmp_min in enumerate(spline_bounds_min):
            kwds = kwds + [{'lower_bound' : tmp_min, 'upper_bound' : spline_bounds_max[i]}]
        bsplines = BSplines(x = df_spline, df = doff, degree = degree , knot_kwds = kwds)
    else:
        bsplines = BSplines(x = df_spline, df = doff, degree = degree)

    ## Return spline model
    return bsplines

def add_spline_vars(df_design, dict_design, df_cov, dict_vars):
    """
    Add columns for spline variables to design dataframe
        - spline basis columns for each spline var (based on Ray's implementation)
    """
    ## Make output dataframe
    df_design_out = df_design.copy()
    
    ## Keep columns that will be included in the final design matrix
    design_vars = df_design.columns.tolist()

    ## Get dict vars
    spline_vars = dict_vars['spline_vars']
    spline_bounds_min = dict_vars['spline_bounds_min']
    spline_bounds_max = dict_vars['spline_bounds_max']

    ## Add spline basis for spline variables
    bsplines = None
    gam_formula = None
    if len(spline_vars) > 0:
        bsplines = calc_spline_model(df_cov[spline_vars], spline_bounds_min, spline_bounds_max)
        df_bsplines = pd.DataFrame(data = bsplines.basis, columns = bsplines.col_names)

        ## Used for fitting the gam model on data with bsplines
        ## Note: Variable names are quoted to allow characters such as ' ' and '-' in var names

        ## FIXME handle this using patsy quoting 
        #gam_formula = 'y ~ ' + ' + '.join(['"' + x + '"' for x in design_vars]) + ' - 1 '

        gam_formula = 'y ~ ' + ' + '.join(design_vars) + ' - 1 '


        df_design_out = pd.concat([df_design_out, df_bsplines], axis = 1)
        design_vars = design_vars + bsplines.col_names
        
        ## Add spline meta data to dict
        dict_design['spline_vars'] = bsplines.col_names
        
        ## Add spline cols to list of non-batch cols
        dict_design['non_batch_vars'] = dict_design['non_batch_vars'] + bsplines.col_names

    ## Return output vars
    return df_design_out[design_vars], dict_design, bsplines, gam_formula

def calc_B_hat(df_data, df_design, dict_batches, bsplines = None, gam_formula = None):
    '''
    Calculate the B hat values
    '''
    #save_to_pickle('/home/guray/sesstmp3.pkl', [df_data, df_design, dict_batches, bsplines, gam_formula])

    ## During estimation print a dot in every "param_dot_count" variables to show the progress
    param_dot_count = 5
    
    ## Get data array
    np_data = np.array(df_data, dtype='float32').T

    ## Get batch info
    n_batch = dict_batches['n_batches']
    n_sample = dict_batches['n_samples']
    sample_per_batch = dict_batches['n_samples_per_batch']

    ## Perform smoothing with GAMs if specified
    np_design = np.array(df_design, dtype='float32')
    
    if bsplines == None:
        B_hat = np.dot(np.dot(np.linalg.inv(np.dot(np_design.T, np_design)), np_design.T), np_data.T)

    else:
        if df_data.shape[1] > 10:
            logger.info('Smoothing more than 10 variables may take several minutes of computation.')

        # initialize penalization weight (not the final weight)
        alpha = np.array([1.0] * bsplines.k_variables)
        
        # initialize an empty matrix for beta
        B_hat = np.zeros((df_design.shape[1], df_data.shape[1]))
        
        # Estimate beta for each variable to be harmonized
        logger.info('Estimating gam model for variables')
        print('   Printing a dot for every variable computed: ')
        df_design_extended = df_design.copy()
        for i, y_var in enumerate(df_data.columns.to_list()):
            
            ## Show progress
            print('.', end = '', flush = True)
            if np.mod(i, param_dot_count) == param_dot_count - 1:
                print('')
            
            ## Add single var to design matrix
            df_design_extended.loc[:, 'y'] = df_data[y_var]
            
            ## Estimate gam model
            gam_bsplines = GLMGam.from_formula(gam_formula, data = df_design_extended, smoother = bsplines, 
                                               alpha = alpha)
            res_bsplines = gam_bsplines.fit()
            
            # Optimal penalization weights alpha can be obtained through gcv/kfold
            # Note: kfold is faster, gcv is more robust

            #gam_bsplines.alpha = gam_bsplines.select_penweight_kfold()[0]

            ## FIXME 
            ##  By default, select_penweight_kfold uses a kfold object with shuffle = True
            ##  This makes it give non-deterministic output
            ##  Current fix :  Call it using an explicit kfold object without shuffling
            ##  To do :  Shuffle data initially once 
            kf = KFold(k_folds = 5, shuffle = False)
            gam_bsplines.alpha = gam_bsplines.select_penweight_kfold(cv_iterator = kf)[0]

            res_bsplines_optim = gam_bsplines.fit()
            
            B_hat[:, i] = res_bsplines_optim.params
        print('\n')

    ## Create B hat dataframe
    df_B_hat = pd.DataFrame(data = B_hat, columns = df_data.columns.tolist(), 
                            index = df_design.columns.tolist())
    
    ## Return B hat dataframe
    return df_B_hat

def standardize_across_features(df_data, df_design, df_B_hat, dict_design, dict_batches):
    '''
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will return all estimated parameters in addition to the
    standardized data.
    '''
    ## Get design columns for batches and non-batches    
    bcol = dict_batches['design_batch_vars']
    nbcol = dict_design['non_batch_vars']
    
    ## Calculate ratio of samples in each batch (to total number of samples)
    r_batches = df_design[bcol].sum() / df_design.shape[0]

    ## Grand mean is the sum of b_hat values for batches weighted by ratio of samples 
    df_grand_mean = df_B_hat.loc[bcol, :].multiply(r_batches, axis = 0).sum(axis = 0)

    ## Data regressed to beta's "for all covars" in the design matrix
    df_fit_all = pd.DataFrame(data = np.dot(df_design, df_B_hat), 
                              columns = df_B_hat.columns, index = df_design.index)

    ## Data regressed to beta's "for non-batch covars" in the design matrix
    df_fit_nb = pd.DataFrame(data = np.dot(df_design[nbcol], df_B_hat.loc[nbcol, :]), 
                             columns = df_B_hat.columns, index = df_design.index)
    
    ## Residuals from the fit
    df_res = df_data - df_fit_all

    ## Calculate pooled var 
    df_var_pooled = (df_res ** 2).mean(axis = 0)

    ## Calculate stand var
    df_stand_mean = df_grand_mean + df_fit_nb
    
    ## Calculate s_data
    df_s_data = (df_data - df_stand_mean).div(np.sqrt(df_var_pooled), axis = 1)

    ### Keep grand_mean and var_pooled in a single dataframe
    df_pooled_stats = pd.concat([df_grand_mean, df_var_pooled], axis = 1).T
    df_pooled_stats.index = ['grand_mean', 'var_pooled']

    return df_s_data, df_stand_mean, df_pooled_stats

#####################################################################################
## Functions specific to HarmonizeToRef


def make_design_dataframe_using_model(df_cov, batch_var, mdl):
    '''
        Expand the covariates dataframe adding columns that will constitute the design matrix
        New columns in the output dataframe are: 
            - one-hot matrix of batch variables (full)
            - one-hot matrix for each categorical var (removing the first column)
            - column for each continuous_vars
            - spline variables are skipped (added later in a separate function)
    '''
    cat_vars = mdl['dict_vars']['cat_vars']
    spline_vars = mdl['dict_vars']['spline_vars']    
    mdl_dict_design = mdl['dict_design']
    mdl_dict_cat = mdl['dict_cat']
    
    ## Make output dataframe
    df_design_out = df_cov.copy()
    
    ## Make output dict
    dict_design = copy.deepcopy(mdl_dict_design)
            
    ## Keep columns that will be included in the final design matrix
    design_vars = []

    ## Add one-hot encoding of batch variable
    df_tmp = pd.get_dummies(df_design_out[batch_var], prefix = batch_var, dtype = float)
    design_batch_vars = df_tmp.columns.tolist()
    df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
    design_vars = design_vars + design_batch_vars

    ## Add numeric variables (from mdl dict_design)
    design_vars = design_vars + mdl_dict_design['num_vars']

    ## Add one-hot encoding for each categoric variable
    ##   If category values match the ones from the mdl, it's simple dummy encoding
    ##   If not: 
    ##     - extend the df with a tmp df with all category values from the mdl;
    ##     - apply dummy encoding
    ##     - remove the tmp df
    for tmp_var in cat_vars:
        
        ## Compare cat values from the mdl to values for curr data
        mdl_tmp_vals = mdl_dict_cat[tmp_var]
        tmp_vals = df_design_out[tmp_var].sort_values().unique().tolist()
        
        if tmp_vals == mdl_tmp_vals:        
            df_tmp = pd.get_dummies(df_design_out[tmp_var], prefix = tmp_var, drop_first = True, dtype = float)
        
        else:
            ## Create a tmp df with cat values from mdl
            df_tmp = pd.DataFrame(data = mdl_tmp_vals, columns = [ tmp_var ])
            
            ## Combine it to cat values for the curr data
            df_tmp = pd.concat([df_tmp, df_design_out[[tmp_var]]])
            
            ## Create dummy cat vars
            df_tmp = pd.get_dummies(df_tmp[tmp_var], prefix = tmp_var, drop_first = True,
                                    dtype = float) 
            
            ## Remove the tmp df values
            df_tmp = df_tmp.iloc[len(mdl_tmp_vals):,:]
            
        df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
        
        design_vars = design_vars + df_tmp.columns.tolist()

    ## Return output vars
    return df_design_out[design_vars], dict_design
    
def update_spline_vars_using_model(df_design, df_cov, mdl):
    '''
        Add columns for spline variables to design dataframe using prev spline mdl
            - spline basis columns for each spline var (based on Ray's implementation)
    '''
    ## Read mdl spline vars
    spline_vars = mdl['dict_vars']['spline_vars']
    bsplines = mdl['bsplines']
    
    ## Make output dataframe
    df_design_out = df_design.copy()

    ## Keep columns that will be included in the final design matrix
    design_vars = df_design.columns.tolist()

    ## Add spline basis for spline variables
    gam_formula = None
    if len(spline_vars) > 0:
        
        ## Existing bspline basis are used to calculate spline columns for the new data
        np_cov_spline = np.array(df_cov[spline_vars])

        ## FIXME handle this using patsy quoting 
        #gam_formula = 'y ~ ' + ' + '.join(['"' + x + '"' for x in design_vars]) + ' - 1 '

        ## Used for fitting the gam mdl on data with bsplines
        gam_formula = 'y ~ ' + ' + '.join(design_vars) + ' - 1 '

        bsplines_basis = bsplines.transform(np_cov_spline)
        df_bsplines = pd.DataFrame(data = bsplines_basis, columns = bsplines.col_names)

        df_design_out = pd.concat([df_design_out, df_bsplines], axis = 1)
        
        design_vars = design_vars + bsplines.col_names

    ## Return output vars
    return df_design_out[design_vars], gam_formula

def standardize_across_features_using_model(df_data, df_design, mdl):
    '''
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    This function will apply a pre-trained harmonization mdl to new data.
    '''
    ## Get mdl data
    mdl_nbcol = mdl['dict_design']['non_batch_vars']
    mdl_var_pooled = mdl['df_pooled_stats'].loc['var_pooled', :]
    mdl_grand_mean = mdl['df_pooled_stats'].loc['grand_mean', :]
    mdl_B_hat = mdl['df_B_hat']

    ## Data regressed to beta's "for non-batch covars" in the design matrix
    ##   Calculated B_hat values from the model
    df_fit_nb = pd.DataFrame(data = np.dot(df_design[mdl_nbcol], mdl_B_hat.loc[mdl_nbcol, :]), 
                             columns = mdl_B_hat.columns, index = df_design.index)

    ## Calculate stand var
    df_stand_mean = mdl_grand_mean + df_fit_nb
    
    ## Calculate s_data
    df_s_data = (df_data - df_stand_mean).div(np.sqrt(mdl_var_pooled), axis = 1)

    return df_s_data, df_stand_mean

def update_model_new_batch(new_batch, mdl_batches, df_gamma_star, df_delta_star):
    '''
    Add estimated batch parameters to the model
    '''
    mdl_batches['batch_values'] = mdl_batches['batch_values'] + [new_batch]
    mdl_batches['df_gamma_star'] = pd.concat([mdl_batches['df_gamma_star'], df_gamma_star], axis=0)
    mdl_batches['df_delta_star'] = pd.concat([mdl_batches['df_delta_star'], df_delta_star], axis=0)
    
    return mdl_batches
