import os
import sys
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import numpy.linalg as la
import copy
from typing import Union, Tuple

import logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
format='%(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)

##logger.setLevel(logging.DEBUG)      ## While debugging
logger.setLevel(logging.INFO)    ## FIXME Debug comments will be removed in release version

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

#import sys
#pwd='/home/guray/Github/neuroHarmonizeV2/neuroHarmonizeV2'
#sys.path.append(pwd)

from .nh_utils import read_data, check_key, make_dict_vars, add_spline_bounds, make_dict_cat, get_data_and_covars, make_dict_batches, make_design_dataframe, add_spline_vars, calc_B_hat, standardize_across_features, fit_LS_model, find_parametric_adjustments, adjust_data_final, calc_aprior, calc_bprior, save_model, save_data

#from nh_utils import read_data, check_key, make_dict_vars, add_spline_bounds, make_dict_cat, get_data_and_covars, make_dict_batches, make_design_dataframe, add_spline_vars, calc_B_hat, standardize_across_features, fit_LS_model, find_parametric_adjustments, adjust_data_final, calc_aprior, calc_bprior, save_model, save_data

##############################
#### FIXME SAVE VARS FOR DEBUG
###import dill;
###dill.dump([df_data, df_design, dict_vars, dict_batches, bsplines, gam_formula], open('./your_bk_dill.pkl', 'wb'));
###if False:
    ###import dill;
    ###[df_data, df_design, dict_vars, dict_batches, bsplines, gam_formula] = dill.load(open('./your_bk_dill.pkl', 'rb'));
##############################

def nh_learn_ref_model(in_data : Union[pd.DataFrame, str], 
                       key_var = None, 
                       batch_var = None, 
                       num_vars = [], 
                       cat_vars = [], 
                       spline_vars = [], 
                       ignore_vars = [],
                       target_vars = [],
                       skip_emp_bayes = False, 
                       out_model_file : str = None,
                       out_data_file : str = None,
                       ) -> Tuple[dict, pd.DataFrame]:
    '''
    Harmonize data and return "the harmonization model", a model that keeps estimated parameters
    for the harmonized reference dataset.
    - This function does not return harmonizated values, but only the model (which can be used 
      to calculate harmonized values by using nh_apply_model());
    
    Arguments
    ---------
    data (REQUIRED): Data to harmonize, in a csv file or pandas DataFrame 
        - Dimensions: n_samples x n_features
        - All columns in in_data that are not labeled as one of key_var, batch_var, cat_vars, 
          num_vars, spline_vars or ignore_vars are considered as data variables that will be corrected.
            
    key_var (OPTIONAL): The primary variable (example: "MRID") (str, should match one of
        in_data columns; if not set, first column is selected as the default primary key)

    batch_var (REQUIRED): The batch variable (example: "Study", or "Site") (str, should match one of
        in_data columns)
        
    num_vars (OPTIONAL): List of numerical variables (list of str, should match items in in_data columns,
        default = [])
        
    cat_vars (OPTIONAL): List of categorical variables (list of str, should match items in in_data columns,
        default = [])
    
    spline_vars (OPTIONAL): List of spline variables (list of str, should match items in in_data columns,
        default = []) 
        - A Generalized Additive Model (GAM) with B-splines is used to calculate a smooth (non-linear) 
        fit for each spline variable

    ignore_vars (OPTIONAL): List of variables to ignore (list of str, should match items in in_data columns,
        default = [])
        
    target_vars (OPTIONAL): List of variables that will be corrected (list of str, should match items in in_data  
        columns; if not set, all columns in in_data other than those listed as a covariate will be considered as data columns)
        
    skip_emp_bayes (OPTIONAL): Whether to skip using empirical Bayes estimates of site effects (bool, 
        default False)

    Returns
    -------
    model : A dictionary of estimated model parameters

        model_ref:  A dictionary with estimated values for the reference dataset
            dict_vars: A dictionary of covariates
            dict_cat: A dictionary of categorical variables and their values
            dict_design: A dictionary of design matrix variables
            bsplines: Bspline model estimated for spline (non-linear) variables
            df_B_hat: Estimated beta parameters for covariates
            df_pooled_stats: Estimated pooled data parameters (grand-mean and pooled variance)
            skip_emp_bayes: Flag to indicate if empirical Bayes was used
                
        model_batches: A dictionary with estimated values for data batches used for harmonization
            batch_values: List of batch values for which harmonization parameters are estimated
            df_gamma_star: Gamma-star (batch-specific location shift) values for each batch
            df_delta_star: Delta-star (batch-specific scaling) values for each batch        
    '''
    
    ## FIXME : fixed seed here for now
    np.random.seed(11)

    ##################################################################
    ## Prepare data

    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    logger.info('Running: nh_learn_ref_model()\n')
    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    logger.info('----------------------- Create Data and Dictionaries -----------------------------')

    logger.info('  Reading input data ...')
    df_in = read_data(in_data)
    if df_in is None:
        sys.exit(1)

    logger.info('  Checking primary key ...')
    key_var = check_key(df_in, key_var)
    if key_var is None:
        sys.exit(1)

    logger.info('  Creating a dictionary of variables ...')
    dict_vars = make_dict_vars(df_in, key_var, batch_var, num_vars, cat_vars, 
                               spline_vars, ignore_vars, target_vars)
    
    #logger.info(dict_vars)
    #input()
    
    
    logger.info('  Adding spline bounds ...')
    dict_vars = add_spline_bounds(df_in, spline_vars, dict_vars) 
    
    logger.info('  Splitting dataframe into covars and data ...')
    res_tmp = get_data_and_covars(df_in, dict_vars)
    if res_tmp is None:
        sys.exit(1)
    else:
        df_cov, df_data = res_tmp

    logger.info('  Creating a dictionary of categorical variables ...')
    dict_cat = make_dict_cat(df_in, cat_vars) 

    ## Create dictionary with batch info
    logger.info('  Creating a dictionary of batch info ...')
    dict_batches = make_dict_batches(df_cov, batch_var)

    ## Create design dataframe      
    logger.info('  Creating the design matrix ...')
    df_design, dict_design = make_design_dataframe(df_cov, dict_vars)    

    ## Add spline terms to the design dataframe
    logger.info('  Adding spline terms to design matrix ...')
    df_design, dict_design, bsplines, gam_formula = add_spline_vars(df_design, dict_design, df_cov, dict_vars)

    ##################################################################
    ## COMBAT Step 1: Standardize dataset

    logger.info('------------------------------ COMBAT STEP 1 ------------------------------')
    
    logger.info('  Calculating B_hat ...')

    df_B_hat = calc_B_hat(df_data, df_design, dict_batches, bsplines, gam_formula)

    logger.info('  Standardizing features ...')
    df_s_data, df_stand_mean, df_pooled_stats = standardize_across_features(df_data, df_design, df_B_hat, dict_design,
                                                                            dict_batches)        
    ##################################################################
    ### COMBAT Step 2: Calculate batch parameters (LS)

    logger.info('------------------------------ COMBAT STEP 2 ------------------------------')

    ##   Step 2.A : Estimate parameters
    logger.info('  Estimating location and scale (L/S) parameters ...')
    dict_LS = fit_LS_model(df_s_data, df_design, dict_batches, skip_emp_bayes)

    ##   Step 2.B : Adjust parameters    
    logger.info('  Adjusting location and scale (L/S) parameters ...')
    df_gamma_star, df_delta_star = find_parametric_adjustments(df_s_data, dict_LS, dict_batches, skip_emp_bayes)        
        
    ##################################################################
    ## COMBAT Step 3: Calculate harmonized data

    logger.info('------------------------------ COMBAT STEP 3 ------------------------------')

    logger.info('  Adjusting final data ...\n')
    df_h_data = adjust_data_final(df_s_data, df_gamma_star, df_delta_star, df_stand_mean, df_pooled_stats,
                                  dict_batches)

    ###################################################################
    ## Prepare output
    
    logger.info('------------------------------ Prepare Output ------------------------------')

    ## Keep B_hat values for batches in a separate dictionary 
    df_B_hat_batches = df_B_hat.loc[dict_batches['design_batch_vars'], :]   
    
    ## In reference dict keep only B_hat values for non-batch variables
    df_B_hat = df_B_hat.loc[dict_design['non_batch_vars'], :]

    ## Create output for the ref model
    mdl_ref = {'dict_vars' : dict_vars, 'dict_cat' : dict_cat, 
               'dict_design' : dict_design, 'bsplines' : bsplines, 'df_B_hat' : df_B_hat,
               'df_pooled_stats' : df_pooled_stats,
               'skip_emp_bayes' : skip_emp_bayes}

    ## Create output for the batches
    #dict_batches = {k:v for k, v in dict_batches.items() if k in ('batch_values', 'design_batch_vars', 
                                                                  #'n_batches')}

    batch_values = dict_batches['batch_values']
    mdl_batches = {'batch_values' : batch_values, 'df_gamma_star' : df_gamma_star, 
                   'df_delta_star' : df_delta_star}

    #mdl_batches = {'df_gamma_star' : df_gamma_star, 
                   #'df_delta_star' : df_delta_star}

    #mdl_batches = {'dict_batches' : dict_batches, 'df_B_hat_batches' : df_B_hat_batches,
                   #'df_gamma_star' : df_gamma_star, 'df_delta_star' : df_delta_star}

    #mdl_batches = {'dict_batches' : dict_batches, 'df_B_hat_batches' : df_B_hat_batches,
                   #'df_gamma_star' : df_gamma_star, 'df_delta_star' : df_delta_star,
                   #'df_gamma_hat' : df_gamma_star, 'df_delta_hat' : df_delta_star,
                   #'dict_LS' : dict_LS}


    ## FIXME : We keep all vars that are not strictly necessary for nh_apply_model()
    ##         in a separate dict (mostly for dubugging purposes for now) 
    mdl_misc = {'df_design' : df_design, 'df_stand_mean' : df_stand_mean, 'df_s_data' : df_s_data,
                'df_B_hat_batches' : df_B_hat_batches,
                'df_gamma_hat' : df_gamma_star, 'df_delta_hat' : df_delta_star,
                'dict_LS' : dict_LS}

    mdl_out = {'mdl_ref' : mdl_ref, 'mdl_batches' : mdl_batches, 'mdl_misc' : mdl_misc}
    #mdl_out = {'mdl_ref' : mdl_ref, 'mdl_batches' : mdl_batches}

    ## Create out dataframe
    param_out_suff = '_HARM'
    df_out = pd.concat([df_cov, df_h_data.add_suffix(param_out_suff)], axis=1)
    
    if out_model_file is not None:
        logger.info('  Saving output model to:\n    ' + out_model_file)
        save_model(mdl_out, out_model_file)

    if out_data_file is not None:
        logger.info('  Saving output data to:\n    ' + out_data_file)
        save_data(df_out, out_data_file)

    logger.info('  Process completed \n')    
    
    return mdl_out, df_out
