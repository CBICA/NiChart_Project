# Gene-SGAN
Gene-SGAN is a multi-view semi-supervised clustering method for disentangling disease heterogeneity. By jointly considering brain phenotypic and genetic data, Gene-SGAN identifies disease subtypes with associated phenotypic and genetic signatures. Using healthy control (HC) populations as a reference distribution, the model effectively clusters participants based on disease-related phenotypic variations with genetic associations, thus avoiding confounders from disease-unrelated factors.


![image info](./datasets/Gene-SGAN.png)

## License
Copyright (c) 2016 University of Pennsylvania. All rights reserved. See[ https://www.cbica.upenn.edu/sbia/software/license.html](https://www.cbica.upenn.edu/sbia/software/license.html)

## Installation
We highly recommend that users install **Anaconda3** on their machines. After installing Anaconda3, Smile-GAN can be used following this procedure:

We recommend that users use the Conda virtual environment:


```bash
$ conda create --name genesgan python=3.8
```
Activate the virtual environment

```bash
$ conda activate genesgan
```
Install GeneSGAN from PyPi:

```bash
$ pip install GeneSGAN
```



## Input structure
The main functions of GeneSGAN basically take three Panda dataframes as data inputs: imaging_data, **imaging_data**, **gene_data**, and **covariate** (optional). Columns with the names *'participant_id'* and *diagnosis* must exist in **imaging_data** and **covariate**. Some conventions for the group label/diagnosis: -1 represents healthy control (HC) and 1 represents patient (PT); categorical variables, such as sex, should be encoded as numbers: Female for 0 and Male for 1, for example. 

Genetic features of all PT but not HC participants in the **imaging_data** need to be provided, so **gene_data** should not have the column *diagnosis*.
The current package only takes SNP data as genetic features, and each SNP variant in **gene_data** needs to be recoded into 0, 1, or 2, indicating the number of minor alleles. 

Example for **imaging_data**:

```bash
participant_id    diagnosis    ROI1    ROI2 ...
subject-1	    -1         325.4   603.4
subject-2            1         260.5   580.3
subject-3           -1         326.5   623.4
subject-4            1         301.7   590.5
subject-5            1	       293.1   595.1
subject-6            1         287.8   608.9
```
Example for **gene_data**:

```bash
participant_id    SNP1    SNP2 ...
subject-2         1       0
subject-4         0       0
subject-5	  2       0
subject-6         0       2
```

Example for **covariate**

```bash
participant_id    diagnosis    age    sex ...
subject-1	    -1         57.3   0
subject-2 	     1         43.5   1
subject-3           -1         53.8   1
subject-4            1         56.0   0
subject-5            1	       60.0   1
subject-6            1         62.5   0
```

## Example
We offer a toy dataset, the ground truth, and the sample code in the folder GeneSGAN/datasets. One-fold training takes around 25 minutes on a MacBook Pro with a 1.4GHz Intel Core i5 and could lead to clustering with around 95% accuracy. A larger fold number could contribute to better clustering performances, so 20 folds or above is recommended in real data applications. Multiple folds can be performed in parallel on HPC clusters.

```bash
import pandas as pd
from GeneSGAN.Gene_SGAN_clustering import cross_validated_clustering, clustering_result

timage_data = pd.read_csv('toy_data_imaging.csv')
gene_data = pd.read_csv('toy_data_gene.csv')
covariate = pd.read_csv('covariate.csv')

output_dir = "PATH_OUTPUT_DIR"
ncluster = 3
start_saving_epoch = 20000
max_epoch = 30000

## three parameters for stopping threshold
WD = 0.12
AQ = 30
cluster_loss = 0.01

## three hyper-parameter to be tuned
genelr = 0.0002
lam = 9
mu = 5
```

When using the package, ***genelr***, ***WD***, ***AQ***, ***cluster\_loss***, ***batch\_size*** need to be chosen empirically:

***genelr***: genelr (i.e., learning rate of the gene step) is the most important hyper-parameter of the model. It is **necessary** to set it to different values, and the value leading to the highest mean N-Asso-SNPs should be used. (**Recommended value**: 0.0004-0.0001).

***WD***: Wasserstein Distance measures the distance between generated PT data along each direction and real PT data. (**Recommended value**: 0.11-0.14)

***AQ***: Alteration Quantity measures the number of participants who change cluster labels during the last three training epochs. Low AQ implies convergence in training. (**Recommended value**: 1/20 of the PT sample size)

***cluster\_loss***: Cluster loss measures how well the clustering function reconstructs the sampled Z variable. (**Recommended value**: 0.01-0.015)

***batch\_size***: Size of the batch for each training epoch. (Default to be 25) It is **necessary** to reset it to 1/8 of the PT sample size.

Some other parameters, ***lam***, ***mu*** have default values but need to be changed in some cases:

***lam***: coefficient controlling the relative importance of cluster\_loss in the training objective function. (Default to be 9).

***mu***: coefficient controlling the relative importance of change\_loss in the training objective function. (Default to be 5).



```bash				    
fold_number = 50  # number of folds the hold-out cv runs
data_fraction = 0.8 # fraction of data used in each fold
cross_validated_clustering(imaging_data, gene_data, ncluster, fold_number, data_fraction, start_saving_epoch, max_epoch,\
					    output_dir, WD, AQ, cluster_loss, genelr = genelr, lam = lam, mu = mu, covariate=covariate)
```

**cross\_validated\_clustering** performs clustering with hold-out cross validation. It is the ***main*** function for clustering. Since the CV process may take a long training time on a normal desktop computer, the function enables an early stop and later resumption. Users can set ***stop\_fold*** to be an early stopping point and ***start\_fold*** depending on the previous stopping point. By setting ***stop\_fold*** to ***start\_fold***+1, users can run multiple iterations **in parellel**, which will significantly reduce the training time.

The function automatically saves a CSV file with clustering results. Two metrics are also provided for hyper-parameter selection: the mean ARI value (i.e., agreements of clusters among all folds) and the mean N-Asso-SNPs (i.e., number of SNPs associated with derived subtypes in test sets).

```					    
model_dirs = ['PATH_TO_CHECKPOINT1','PATH_TO_CHECKPOINT2',...] #list of paths to previously saved checkpoints (with name 'converged_model_foldk' after cv process)
cluster_label, cluster_probabilities, _, _, _, _ = clustering_result(model_dirs, ncluster, imaging_data, gene_data, covariate = covariate)
```
**clustering\_result** is a function used for clustering patient data using previously saved models. **imaging_data** and **covariate** (optional) should be Panda dataframes with the same format as introduced before. Only PT data (which can be inside or outside of the training set) for which the user wants to derive subtype memberships needs to be provided with diagnoses set to 1. **gene_data** is not required when applying the trained models. ***The function returns cluster labels of PT data following the order of PT in the provided dataframe.***


## Citation
If you use this package for research, please cite the following paper:


```bash
@misc{yang2023genesgan,
  doi = {10.48550/ARXIV.2301.10772},
  url = {https://arxiv.org/abs/2301.10772},
  author = {Yang, Zhijian and Wen, Junhao and Abdulkadir, Ahmed and Cui, Yuhan and Erus, Guray and Mamourian, Elizabeth and Melhem, Randa and Srinivasan, Dhivya and Govindarajan, Sindhuja T. and Chen, Jiong and Habes, Mohamad and Masters, Colin L. and Maruff, Paul and Fripp, Jurgen and Ferrucci, Luigi and Albert, Marilyn S. and Johnson, Sterling C. and Morris, John C. and LaMontagne, Pamela and Marcus, Daniel S. and Benzinger, Tammie L. S. and Wolk, David A. and Shen, Li and Bao, Jingxuan and Resnick, Susan M. and Shou, Haochang and Nasrallah, Ilya M. and Davatzikos, Christos},  
  title = {Gene-SGAN: a method for discovering disease subtypes with imaging and genetic signatures via multi-view weakly-supervised deep clustering},  
  publisher = {arXiv},  
  year = {2023},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```


