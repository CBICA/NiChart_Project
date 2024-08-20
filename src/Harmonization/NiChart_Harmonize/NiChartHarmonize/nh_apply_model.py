import logging
import os
import sys
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import numpy.linalg as la
import copy
from typing import Union, Tuple

import pickle
#import dill

import logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)

#logger.setLevel(logging.DEBUG)      ## While debugging
logger.setLevel(logging.INFO)    ## FIXME Comments will be removed in release version

#import sys
#pwd='/home/guray/Github/neuroHarmonizeV2/neuroHarmonizeV2'
#sys.path.append(pwd)

from .nh_utils import read_data, read_model, verify_data_to_model, check_key, filter_data, get_data_and_covars, make_dict_vars, make_dict_batches, make_design_dataframe_using_model, update_spline_vars_using_model, standardize_across_features_using_model, update_model_new_batch, fit_LS_model, find_parametric_adjustments, adjust_data_final, calc_aprior, calc_bprior, save_model, save_data

#from nh_utils import read_data, read_model, verify_data_to_model, check_key, filter_data, get_data_and_covars, make_dict_vars, make_dict_batches, make_design_dataframe_using_model, update_spline_vars_using_model, standardize_across_features_using_model, update_model_new_batch, fit_LS_model, find_parametric_adjustments, adjust_data_final, calc_aprior, calc_bprior, save_model, save_data

#from nh_utils import fitLSModelAndFindPriorsV2

## FIXME Example to save curr vars
#print('SAVING')
#fname = 'sesstmp1.pkl'
#save_to_pickle(fname, [df_cov, batch_var, model])


def nh_harmonize_to_ref(in_data : Union[pd.DataFrame, str],
                        in_model : Union[dict, str],
                        ignore_saved_batch_params = False,
                        out_model_file : str = None,
                        out_data_file : str = None
                        ) -> Tuple[dict, pd.DataFrame]:

    '''
    Harmonize each batch in the input dataset to the reference data (the input model):
    - Existing batches (in-sample): use saved parameters in the model;
    - New batches (out-of-sample): calculate parameters for the new batch and update the model
    
    Arguments
    ---------
    df_data (REQUIRED): Data to harmonize, in a pandas DataFrame 
        - Dimensions: n_samples x n_features
    
    df_cov (REQUIRED): Covariates, in a pandas DataFrame 
        - Dimensions: n_samples x n_covars;
        Columns in df_cov should match with harmonization covariates that were used for 
        calculating the input model

    ignore_saved_batch_params (OPTIONAL): Flag to ignore saved values for all batches and 
        recalculate them  (True or False, default = False)

    Returns
    -------
    model: Updated model. Parameters estimated for new batch(es) are added to the model_batches 
        dictionary
    
    df_out: Output dataframe with input covariates and harmonized variables
    '''
    
    ## FIXME : fixed seed here for now
    np.random.seed(11)

    ##################################################################
    ## Prepare data

    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    logger.info('Running: nh_harmonize_to_ref()\n')    
    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    logger.info('--------------------------- Read Data and Model ---------------------------')

    logger.info('  Reading input data ...')
    df_in = read_data(in_data)
    if df_in is None:
        sys.exit(1)

    logger.info('  Reading input model ...')
    mdl_in = read_model(in_model)
    if mdl_in is None:
        sys.exit(1)
    mdl_out = mdl_in

    logger.info('  Verify data against model ...')
    missing_vars = verify_data_to_model(mdl_in, df_in)
    logger.info(len(missing_vars))
    ##input()
    
    if len(missing_vars) > 0:
        logger.info('Data does not match model, missing columns: ' + missing_vars)
        return missing_vars

    ## Read data fields from model
    mdl_batches = mdl_in['mdl_batches']
    mdl_ref = mdl_out['mdl_ref']
    dict_vars = mdl_ref['dict_vars']
    dict_cat = mdl_ref['dict_cat']
    batch_var = dict_vars['batch_var']

    #logger.info(dict_vars)
    #input()
    
    logger.info('  Checking primary key ...')
    key_var = check_key(df_in, dict_vars['key_var'])
    if key_var is None:
        sys.exit(1)
        

    logger.info('  Filtering data ...')
    df_in = filter_data(df_in, dict_cat)

    logger.info('  Splitting dataframe into covars and data ...')
    res_tmp = get_data_and_covars(df_in, dict_vars)
    if res_tmp is None:
        sys.exit(1)
    else:
        df_cov, df_data = res_tmp
    
    ##################################################################
    ## Harmonize each batch individually
    
    ## Output df
    df_h_data = pd.DataFrame(index = df_data.index, columns = df_data.columns)
    
    batch_var_vals = df_cov[batch_var].unique()
    for curr_batch in batch_var_vals:

        logger.info('Harmonizing batch : ' + str(curr_batch))
        
        ##################################################################
        ## Prepare data for the current batch
        
        ## Select batch
        ind_curr = df_cov[df_cov[batch_var] == curr_batch].index.tolist() 
                
        df_cov_curr = df_cov.loc[ind_curr, :].copy().reset_index(drop = True)
        df_data_curr = df_data.loc[ind_curr, :].copy().reset_index(drop = True)

        logger.info('  ------------------------------ Prep Data -----------------------------')
    
        ## Create dictionary with batch info
        logger.info('    Creating batches dictionary ...')
        dict_batches = make_dict_batches(df_cov_curr, batch_var)
        
        ## Create design dataframe
        logger.info('    Creating design matrix ...')            
        df_design, dict_design = make_design_dataframe_using_model(df_cov_curr, batch_var, mdl_ref)    

        ## Add spline terms to design dataframe
        logger.info('    Adding spline terms to design matrix ...')
        df_design, gam_formula = update_spline_vars_using_model(df_design, df_cov_curr, mdl_ref)

        ##################################################################
        ## COMBAT Step 1: Standardize dataset

        logger.info('  ------------------------------ COMBAT STEP 1 ------------------------------')
        
        logger.info('    Standardizing features ...')
        df_s_data, df_stand_mean = standardize_across_features_using_model(df_data_curr, df_design, mdl_ref)

        ##################################################################
        ##   ----------------  IN-SAMPLE HARMONIZATION ----------------
        ##   Current batch is one of the batches used for creating the ref model
        ##   - Skip Combat Step 2 (estimation of LS parameters)
        ##   - Use previously estimated model parameters to align new batch to reference model
        
        if curr_batch in mdl_batches['batch_values']:
            
            logger.info('  Batch in model; running IN-SAMPLE ...')

            ##################################################################
            ## COMBAT Step 3: Calculate harmonized data

            logger.info('  ------------------------------ COMBAT STEP 3 ------------------------------')
            
            ## Calculate harmonized data
            logger.info('    Adjusting final data ...\n')
            df_h_data_curr = adjust_data_final(df_s_data, mdl_batches['df_gamma_star'],
                                               mdl_batches['df_delta_star'],
                                               df_stand_mean, mdl_ref['df_pooled_stats'], dict_batches)
                        
        ##################################################################
        ##   ----------------  OUT-OF-SAMPLE HARMONIZATION ----------------
        ##   Current batch is not one of the batches used for creating the ref model
        ##   - Estimate parameters to align new batch to the reference model
        ##   - Save estimated parameters (update the model)
        ##   - Return harmonized data and updated model
        
        else:
            
            logger.info('  Batch not in model; running OUT-OF-SAMPLE ...')
            
            ##################################################################
            ### COMBAT Step 2: Calculate batch parameters (LS)

            logger.info('  ------------------------------ COMBAT STEP 2 ------------------------------')

            ##   Step 2.A : Estimate parameters
            logger.info('    Estimating location and scale (L/S) parameters ...')            
            dict_LS = fit_LS_model(df_s_data, df_design, dict_batches, mdl_ref['skip_emp_bayes'])

            ##   Step 2.B : Adjust parameters    
            logger.info('    Adjusting location and scale (L/S) parameters ...')
            df_gamma_star, df_delta_star = find_parametric_adjustments(df_s_data, dict_LS,
                                                                       dict_batches, 
                                                                       mdl_ref['skip_emp_bayes'])

            ##################################################################
            ## COMBAT Step 3: Calculate harmonized data

            logger.info('  ------------------------------ COMBAT STEP 3 ------------------------------')

            logger.info('    Adjusting final data ...\n')
            df_h_data_curr = adjust_data_final(df_s_data, df_gamma_star, df_delta_star, 
                                          df_stand_mean, mdl_ref['df_pooled_stats'], dict_batches)

            ###################################################################
            ## Update model
            
            logger.info('  ------------------------------ Update Model ------------------------------')

            logger.info('    Updating model with new batch ...\n')
            mdl_batches = update_model_new_batch(curr_batch, mdl_batches, df_gamma_star, df_delta_star)            

        ###################################################################
        ## Update harmonized data df

        logger.info('  ------------------------------ Update h_data  ------------------------------')
        df_h_data.loc[ind_curr, :] = df_h_data_curr.values
        
    ###################################################################
    ## Prepare output
    
    logger.info('------------------------------ Prepare Output ------------------------------')
    
    ## Set updated batches in output model (This should be the only change in the model)
    mdl_out['mdl_batches'] = mdl_batches
    
    ## Create out dataframe
    param_out_suff = '_HARM'    
    df_out = pd.concat([df_cov, df_h_data.add_suffix(param_out_suff)], axis=1)

    ###################################################################
    ## Return output
    if out_model_file is not None:
        logger.info('  Saving output model to:\n    ' + out_model_file)
        save_model(mdl_out, out_model_file)

    if out_data_file is not None:
        logger.info('  Saving output data to:\n    ' + out_data_file)
        save_data(df_out, out_data_file)

    logger.info('  Process completed \n')    

    return mdl_out, df_out
    
