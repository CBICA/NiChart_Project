## GenFAR


The development of deep learning models in medical imaging has remained relatively fragmented and specialized to individual use cases, limiting their generalizability and clinical utility. Here we present GenFAR, a modular deep learning framework that learns general, clinically informed features from T1-weighted brain MRI. We trained this modular architecture on 49,246 individuals across 11 cohorts, using 17 diverse tasks spanning cognition, diagnosis, demographics, and biomarkers. This yields aggregated, focused feature sets that capture rich, clinically-relevant brain representations. We developed two complementary frameworks: independent parallel channels and a sequential learning approach where tasks progressively build on previously learned representations. Through an analysis of 5,000 task sequences, we identify an optimal sequence length of six tasks and introduce a Donor Score metric to quantify each task's contribution to downstream performance. This analysis reveals five consistently strong donor tasks (Age, AD/MCI, MMSE, Hypertension, Hyperlipidemia) that form the base of our sequential model. We demonstrate the utility of our learned representation, in various tasks beyond those included in the training set, to serve as the foundation for specialized secondary predictors. We further show that using the learned feature representation can substantially increase the sample efficiency of secondary deep learning training tasks and models, as well as improve their accuracy.


#### Input:

- T1-weighted scans (Nifti)

#### Output

- GenFAR loadings (csv file)
