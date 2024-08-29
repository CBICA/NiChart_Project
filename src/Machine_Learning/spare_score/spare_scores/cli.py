import argparse

import pkg_resources  # type: ignore

from .spare import spare_test, spare_train

VERSION = pkg_resources.require("spare_scores")[0].version


def main() -> None:

    prog = "spare_scores"
    description = "SPARE model training & scores calculation"
    usage = """
    spare_scores  v{VERSION}.
    SPARE model training & scores calculation
    required arguments:
        [ACTION]        The action to be performed, either 'train' or 'test'
        [-a, --action]
        [INPUT]         The dataset to be used for training / testing. Can be
        [-i, --input]   a filepath string of a .csv file.
    optional arguments:
        [OUTPUT]        The filename for the model (as a .pkl.gz) to be saved
        [-o, --output]  at, if training. If testing, the filepath of the
                        resulting SPARE score dataframe (as a .csv file) to be
                        saved. If not given, nothing will be saved.
        [MODEL]         The model to be used (only) for testing. Can be a
        [-m, --model,   filepath string of a .pkl.gz file. Required for testing
        --model_file]
        [KEY_VAR]       The key variable to be used for training. This could
        [-kv,           be a string of a column name that can uniquely
        --key_var,      identify a row of the dataset.
        --identifier]   For example (if a row_ID doesn't exist), it could be:
                        --key_var PTID
                        If not given, the first column of the dataset is
                        considered the primary key of the dataset. Required for
                        training.
        [DATA_VARS]     The list of predictors to be used for training. List.
        [-dv,           If not given, training will assume that all (apart from
        --data_vars,    the key variables) variables will be used as
        --predictors]   predictors, with the ignore variables ignored.
        [IGNORE_VARS]   The list of predictors to be ignored for training. Can
        [-iv,           be a list, or empty.
        --ignore_vars,
        --ignore]
        [TARGET]        The characteristic to be predicted in the course of the
        [-t,            training. String of the name of the column. Required
        --target,       for training.
        --to_predict]
        [POS_GROUP]     Group to assign a positive SPARE score (only for
        -pg,            classification). String. Required for training.
        --pos_group]
        [MODEL_TYPE]    The type of model to be used for training. String.
        [-mt,           'SVM', 'MLP' 'MLPTorch'. Required for training.
        --model_type]
        [KERNEL]        The kernel for SVM training. 'linear' or 'rbf' (only
        -k,             linear is supported currently in regression).
        --kernel]
        [SPARE_VAR]     The name of the column to be used for SPARE score. If
        [-sv,           not given, the column will be named 'SPARE_score'.
        --spare_var]
        [VERBOSE]       Verbosity. Int.
        [-v,            0: Warnings
        --verbose,      1: Info
        --verbosity]    2: Debug
                        3: Errors
                        4: Critical
        [LOGS]          Where to save log file. If not given, logs will be
        [-l,            printed out.
        --logs]
        [VERSION]       Display the version of the package.
        [-V, --version]
        [HELP]          Show this help message and exit.
        [-h, --help]
    """.format(
        VERSION=VERSION
    )

    parser = argparse.ArgumentParser(
        prog=prog, usage=usage, description=description, add_help=False
    )

    # ACTION argument
    help = "The action to be performed, either 'train' or 'test'"
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        help=help,
        choices=["train", "test"],
        default=None,
        required=True,
    )

    # INPUT argument
    help = (
        "The dataset to be used for training / testing. Can be"
        + "a filepath string of a .csv file."
    )
    parser.add_argument(
        "-i", "--input", type=str, help=help, default=None, required=True
    )

    # OUTPUT argument
    help = (
        "The filename for the model (as a .pkl.gz) to be saved "
        + "at, if training. If testing, the filepath of the "
        + "resulting SPARE score dataframe (as a .csv file) to be "
        + "saved. If not given, nothing will be saved."
    )
    parser.add_argument(
        "-o", "--output", type=str, help=help, default=None, required=False
    )

    # MODEL argument
    help = (
        "The model to be used (only) for testing. Can be a "
        + "filepath string of a .pkl.gz file. Required for testing."
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model_file",
        type=str,
        help=help,
        default=None,
        required=False,
    )

    # KEY_VAR argument
    help = (
        "The key variable to be used for training. This could "
        + "be a string of a column name that can uniquely "
        + "identify a row of the dataset. "
        + "For example (if a row_ID doesn't exist), it could be: "
        + "--key_var PTID"
        + "If not given, the first column of the dataset is "
        + "considered the primary key of the dataset. Required for"
        + "training."
    )
    parser.add_argument(
        "-kv", "--key_var", "--identifier", type=str, default="", required=False
    )

    # DATA_VARS argument
    help = (
        "The list of predictors to be used for training. List. "
        + "If not given, training will assume that all (apart from "
        + "the key variables) variables will be used as "
        + "predictors, with the ignore variables ignored."
    )
    parser.add_argument(
        "-dv",
        "--data_vars",
        "--predictors",
        type=str,
        nargs="+",
        default=[],
        required=False,
    )

    # IGNORE_VARS argument
    help = (
        "The list of predictors to be ignored for training. Can be a list,"
        + " or empty."
    )
    parser.add_argument(
        "-iv",
        "--ignore_vars",
        "--ignore",
        type=str,
        nargs="+",
        default=[],
        required=False,
    )

    # TARGET argument
    help = (
        "The characteristic to be predicted in the course of the "
        + "training. String of the name of the column. Required "
        + "for training."
    )
    parser.add_argument(
        "-t",
        "--target",
        "--to_predict",
        type=str,
        help=help,
        default=None,
        required=False,
    )

    # POS_GROUP argument
    help = (
        "Group to assign a positive SPARE score (only for classification)."
        + " String. Required for training."
    )
    parser.add_argument(
        "-pg", "--pos_group", type=str, help=help, default=None, required=False
    )

    # MODEL_TYPE argument
    help = (
        "The type of model to be used for training. String. "
        + "'SVM' or 'MLP'. Required for training."
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        help=help,
        choices=["SVM", "MLP", "MLPTorch"],
        default="SVM",
        required=False,
    )

    # KERNEL argument
    help = (
        "The kernel for the training. 'linear' or 'rbf' (only linear is "
        + "supported currently in regression)."
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["linear", "rbf"],
        help=help,
        default="linear",
        required=False,
    )

    # SPARE_VAR argument
    help = (
        "The name of the column to be used for SPARE score. If not given, "
        + "the column will be named 'SPARE_score'."
    )
    parser.add_argument(
        "-sv", "--spare_var", type=str, help=help, default="SPARE_score", required=False
    )

    # VERBOSE argument
    help = "Verbose"
    parser.add_argument(
        "-v", "--verbose", "--verbosity", type=int, help=help, default=1, required=False
    )

    # LOGS argument
    help = "Where to save log file. If not given, logs will only be printed " + "out."
    parser.add_argument(
        "-l", "--logs", type=str, help=help, default=None, required=False
    )

    # VERSION argument
    help = "Show the version and exit"
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=prog + ": v{VERSION}.".format(VERSION=VERSION),
        help=help,
    )

    # HELP argument
    help = "Show this message and exit"
    parser.add_argument("-h", "--help", action="store_true", help=help)

    arguments = parser.parse_args()

    if arguments.action == "train":
        if arguments.target is None:
            print(usage)
            print("The following argument is required: -t/--target" + "/--to_predict")
            return

        spare_train(
            arguments.input,
            arguments.target,
            arguments.model_type,
            arguments.pos_group,
            arguments.key_var,
            arguments.data_vars,
            arguments.ignore_vars,
            arguments.kernel,
            arguments.output,
            arguments.verbose,
            arguments.logs,
        )
        return

    if arguments.action == "test":
        if arguments.model is None:
            print(usage)
            print("The following arguments are required: -m/--model/" + "--model_file")
            return

        spare_test(
            arguments.input,
            arguments.model,
            arguments.key_var,
            arguments.output,
            arguments.spare_var,
            arguments.verbose,
            arguments.logs,
        )
        return

    return
