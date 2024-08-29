import sys
import argparse
import pandas as pd
from typing import Tuple, Union
from .nh_learn_model import nh_learn_ref_model
from .nh_apply_model import nh_harmonize_to_ref

import logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
format='%(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)

##logger.setLevel(logging.DEBUG)      ## While debugging
logger.setLevel(logging.INFO)    ## FIXME Debug comments will be removed in release version

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


def main():
    prog="neuroharm"
    description = "Harmonization learn ref model & apply to new data"
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)

    # Action
    help = "The action to be performed, either 'learn' or 'apply'"
    parser.add_argument("-a", 
                        "--action", 
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)
    
    # Data file
    help = "Data file to be used as input"
    parser.add_argument("-i",
                        "--in_data_file", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model file
    help = "The input model file (only for action: apply)"
    parser.add_argument("-m",
                        "--in_model_file",
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Key variable
    help = "Primary key of the data file. If not provided, the first column is considered as the primary key"
    parser.add_argument("-k",
                        "--key_var", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Batch variable
    help = "Batch variable (e.g. site, study, scanner)"
    parser.add_argument("-b",
                        "--batch_var", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Numeric variables
    help = "Numeric covariates that will be modeled using a linear model"
    parser.add_argument("-n",
                        "--num_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Categorical variables
    help = "Categoric covariates"
    parser.add_argument("-c",
                        "--cat_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Spline variables
    help = "Numeric covariates that will be modeled using a spline model"
    parser.add_argument("-s",
                        "--spline_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)
    
    # Variables to ignore
    help = "Variables that will be dropped / ignored"
    parser.add_argument("-g",
                        "--ignore_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Data variables
    help = "Variables that will be harmonized. If not provided, all variables in the input data file after removing other listed covariates will be used as data variables"
    parser.add_argument("-t",
                        "--target_vars",
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Flag to run / bypass empirical Bayes estimation
    help = "Flag to skip empirical Bayes"
    parser.add_argument("-e",
                        "--skip_emp_bayes", 
                        action = 'store_true',
                        help=help, 
                        required=False)

    # Output file
    help = "File name to save the output model"
    parser.add_argument("-o",
                        "--out_model_file", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Output data
    help = "File name to save the output data"
    parser.add_argument("-u",
                        "--out_data_file", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Verbosity argument
    help = "Verbosity"
    parser.add_argument("-v", 
                        type=int, 
                        help=help, 
                        default=1, 
                        required=False)
        
    args = parser.parse_args()
    
    ### Print args
    #print(args)
    #print('aaa')
    #input()
    
    ## Verify required args conditional to selected action
    if args.action == 'learn':
        if args.batch_var == None:
            logger.error('Missing required arg: -b/--batch_var')
            sys.exit(1)
    
    
    ## Call harmonize functions
    if args.action == 'learn':
        try: 
            mdlOut, dfOut = nh_learn_ref_model(args.in_data_file, 
                                            args.key_var, 
                                            args.batch_var,
                                            args.num_vars, 
                                            args.cat_vars, 
                                            args.spline_vars, 
                                            args.ignore_vars, 
                                            args.target_vars,                                            
                                            args.skip_emp_bayes, 
                                            args.out_model_file, 
                                            args.out_data_file)
        except:
            logger.error('Failed ...')
    if args.action == 'apply':
        try:
            mdlOut, dfOut = nh_harmonize_to_ref(args.in_data_file, 
                                            args.in_model_file,
                                            ignore_saved_batch_params = False,
                                            out_model_file = args.out_model_file,
                                            out_data_file = args.out_data_file)
        except:
            logger.error('Failed ...')
    
    return;
