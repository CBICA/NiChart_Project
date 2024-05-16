
# Compute SPARE Scores for Your Case

"SPARE" is short for "Spatial Pattern of Abnormalities for Recognition of ..." If you have brain images of a case population, such as the Alzheimer's disease (AD), the SPARE model will try to find characteristic brain patterns of AD with respect to a control population, such as cognitively normal. This would be an example of a classification-based SPARE model (currently powered by support vector machine or SVM). This model (that we named SPARE-AD) then computes SPARE-AD scores on an individual-basis that indicates how much the individual carries the learned brain patterns of AD.

Alternatively, you may want to find the spatial pattern related to brain aging (BA). In this case, you would provide sample images and indicate that chronological age is what you expect the model to learn patterns for. This would be an example of a regression-based SPARE model (also powered by SVM). This model (that we named SPARE-BA) then computes SPARE-BA scores on an individual-basis that predicts your brain age.
\
\
\
For detailed documentation, please see here: **[spare_scores](https://cbica.github.io/spare_score/)**

## Installation

### Conda environment using pip

```bash
    conda create -n spare python=3.8
    conda activate spare
    conda install pip
    pip install spare_scores
```

### Python3 virtual environment using pip

```bash
    python3 -m venv env spare
    source spare/bin/activate
    pip install spare_scores
```

### Conda environment from Github repository

```bash
    git clone https://github.com/CBICA/spare_score.git
    cd spare_score
    pip install .
```

## Usage

```text
spare_scores  v1.0.0.
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
        [-mt,           'SVM' or 'MLP'. Required for training.
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
```

## Examples

Example of training a model (given the example data):

```bash
spare_score --action train \
            --input spare_scores/data/example_data.csv \
            --predictors H_MUSE_Volume_11 H_MUSE_Volume_23 H_MUSE_Volume_30 \
            --ignore_vars Sex \
            --to_predict Age \
            --kernel linear \
            --verbose 2 \
            --output my_model.pkl.gz
```

Example of testing (applying) a model (given the example data):

```bash
spare_score -a test \
            -i spare_scores/data/example_data.csv  \
            --model my_model.pkl.gz \
            -o test_spare_data.csv \
            -v 0 \
            --logs test_logs.txt
```

## References

- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

## Disclaimer

- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- By using spare_scores, the user agrees to the following license: [CBICA Software License](https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html)

## Contact

For more information and support, please post on the [Discussions](https://github.com/CBICA/spare_score/discussions) section or contact [CBICA Software](mailto:software@cbica.upenn.edu)
