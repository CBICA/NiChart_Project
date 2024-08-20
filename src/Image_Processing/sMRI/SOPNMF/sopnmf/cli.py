import argparse

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def opnmf_func(args):
    """
    The default function to run opNMF.
    Args:
        args: args from parser

    Returns:

    """
    from sopnmf.opnmf_core import opnmf
    opnmf(
        args.participant_tsv,
        args.output_dir,
        args.tissue_binary_mask,
        args.num_component_min,
        args.num_component_max,
        args.num_component_step,
        args.max_iter,
        args.magnitude_tolerance,
        args.early_stopping_epoch,
        args.n_threads,
        args.verbose
    )

def sopnmf_func(args):
    """
    The default function to run sopNMF.
    Args:
        args: args from parser

    Returns:

    """
    from sopnmf.opnmf_core import sopnmf
    sopnmf(
        args.participant_tsv,
        args.participant_tsv_max_memory,
        args.output_dir,
        args.tissue_binary_mask,
        args.num_component_min,
        args.num_component_max,
        args.num_component_step,
        args.batch_size,
        args.max_iter,
        args.magnitude_tolerance,
        args.early_stopping_epoch,
        args.n_threads,
        args.verbose
    )

def apply_to_train_func(args):
    """
    The default function to apply the trained model to the training data.
    Args:
        args: args from parser

    Returns:

    """
    from sopnmf.opnmf_post import apply_to_training
    apply_to_training(
        args.output_dir,
        args.num_component,
        args.tissue_binary_mask,
        args.output_suffix,
        args.verbose
    )

def apply_to_test_func(args):
    """
    The default function to apply the trained model to the unseen test data.
    Args:
        args: args from parser

    Returns:

    """
    from sopnmf.opnmf_post import apply_to_test
    apply_to_test(
        args.output_dir,
        args.num_component,
        args.tissue_binary_mask,
        args.participant_tsv,
        args.output_suffix,
        args.verbose
    )

def parse_command_line():
    """
    Definition for the commandline parser
    Returns:

    """

    parser = argparse.ArgumentParser(
        prog='sopnmf',
        description='Orthogonal Projective Non-negative Matrix Factorization for interpretable neuroimaging analysis.')

    subparser = parser.add_subparsers(
        title='''Algorithm to run per needs:''',
        description='''Which algorithm do you want to use with OPNMF?
            (sopnmf, sopnmf, apply_train, apply_test).''',
        dest='algorithm',
        help='''****** Algorithm proposed by OPNMF ******''')

    subparser.required = True

########################################################################################################################
### opNMF
########################################################################################################################
    opnmf_func_parser = subparser.add_parser(
        'opnmf',
        help='Perform OPNMF algorithm.')

    opnmf_func_parser.add_argument(
        'participant_tsv',
        help="Path to the tsv containing participant information. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the path. ",
        default=None
    )

    opnmf_func_parser.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None, type=str
    )

    opnmf_func_parser.add_argument(
        'tissue_binary_mask',
        help='This is a tissue binary mask that constrains the voxels only in desired regions.',
        default=None, type=str
    )

    opnmf_func_parser.add_argument(
        '-min', '--num_component_min',
        help='The minimum number of components',
        type=int, default=None
    )

    opnmf_func_parser.add_argument(
        '-max', '--num_component_max',
        help='The maximum number of components',
        type=int, default=None
    )

    opnmf_func_parser.add_argument(
        '-step', '--num_component_step',
        help='Step size for number of components',
        type=int, default=1
    )

    opnmf_func_parser.add_argument(
        '-mi', '--max_iter',
        help='Maximum number of iterations for convergence',
        type=int, default=50000
    )

    opnmf_func_parser.add_argument(
        '-mt', '--magnitude_tolerance',
        help='The tolerance of loss change magnitude for early stopping',
        type=float, default=0
    )

    opnmf_func_parser.add_argument(
        '-ese', '--early_stopping_epoch',
        help='The tolerance of number of bad epochs for early stopping',
        type=int, default=20
    )

    opnmf_func_parser.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    opnmf_func_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    opnmf_func_parser.set_defaults(func=opnmf_func)

########################################################################################################################
### sopNMF
########################################################################################################################
    sopnmf_func_parser = subparser.add_parser(
        'sopnmf',
        help='Perform stochastic version of opNMF algorithm - sopNMF.')

    sopnmf_func_parser.add_argument(
        'participant_tsv',
        help="Path to the tsv containing participant information. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the path. ",
        default=None
    )

    sopnmf_func_parser.add_argument(
        'participant_tsv_max_memory',
        help="Path to the tsv containing subpopulation of the participants. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the path. ",
        default=None
    )

    sopnmf_func_parser.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None, type=str
    )

    sopnmf_func_parser.add_argument(
        'tissue_binary_mask',
        help='This is a tissue binary mask that constrains the voxels only in desired regions.',
        default=None, type=str
    )

    sopnmf_func_parser.add_argument(
        '-min', '--num_component_min',
        help='The minimum number of components',
        type=int, default=None
    )

    sopnmf_func_parser.add_argument(
        '-max', '--num_component_max',
        help='The maximum number of components',
        type=int, default=None
    )

    sopnmf_func_parser.add_argument(
        '-step', '--num_component_step',
        help='Step size for number of components',
        type=int, default=1
    )

    sopnmf_func_parser.add_argument(
        '-bs', '--batch_size',
        help='Batch size for the stochastic Lagrangian update rules',
        type=int, default=8
    )

    sopnmf_func_parser.add_argument(
        '-mi', '--max_iter',
        help='Maximum number of iterations for convergence',
        type=int, default=50000
    )

    sopnmf_func_parser.add_argument(
        '-mt', '--magnitude_tolerance',
        help='The tolerance of loss change magnitude for early stopping',
        type=float, default=0
    )

    sopnmf_func_parser.add_argument(
        '-ese', '--early_stopping_epoch',
        help='The tolerance of number of bad epochs for early stopping',
        type=int, default=20
    )

    sopnmf_func_parser.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    sopnmf_func_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    sopnmf_func_parser.set_defaults(func=sopnmf_func)


########################################################################################################################
### apply_to_training
########################################################################################################################
    apply_to_training_func_parser = subparser.add_parser(
        'apply_to_training',
        help='Apply the training data to the trained model.')

    apply_to_training_func_parser.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None, type=str
    )

    apply_to_training_func_parser.add_argument(
        'num_component',
        help='Number of components',
        type=int, default=None
    )

    apply_to_training_func_parser.add_argument(
        'tissue_binary_mask',
        help='This is a tissue binary mask that constrains the voxels only in desired regions.',
        default=None, type=str
    )

    apply_to_training_func_parser.add_argument(
        '-os', '--output_suffix',
        help='The suffix to add to the output tsv files',
        default=None, type=str
    )

    apply_to_training_func_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    apply_to_training_func_parser.set_defaults(func=apply_to_train_func)

########################################################################################################################
### apply_to_test
########################################################################################################################
    apply_to_test_func_parser = subparser.add_parser(
        'apply_to_test',
        help='Apply the test data to the trained model.')

    apply_to_test_func_parser.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None, type=str
    )

    apply_to_test_func_parser.add_argument(
        'num_component',
        help='Number of components',
        type=int, default=None
    )

    apply_to_test_func_parser.add_argument(
        'tissue_binary_mask',
        help='This is a tissue binary mask that constrains the voxels only in desired regions.',
        default=None, type=str
    )

    apply_to_test_func_parser.add_argument(
        'participant_tsv',
        help="Path to the tsv containing participant information. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the path. ",
        default=None
    )

    apply_to_test_func_parser.add_argument(
        '-os', '--output_suffix',
        help='The suffix to add to the output tsv files',
        default=None, type=str
    )

    apply_to_test_func_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    apply_to_test_func_parser.set_defaults(func=apply_to_test_func)

    return parser











