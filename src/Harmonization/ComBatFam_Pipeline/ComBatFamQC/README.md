# ComBatFamQC

The **ComBatFamQC** package is a powerful tool designed to streamline interactive batch effect diagnostics, harmonization, and post-harmonization downstream analysis. This package is specifically tailored to provide both interactive qualitative visualization and statistical testing for batch effects diagnostics, as well as to offer various easily-used built-in harmonization techniques to facilitate a better harmonization process.

Additionally, the package provides life span age trends of brain structures and residual datasets, eliminating specific covariates' effects to better conduct post-harmonization downstream analysis. In terms of the final delivery, it will provide interactive visualization through R Shiny for batch effect diagnostics and age trend visualization. Additionally, it integrates the harmonization process and can provide a harmonized dataset, fitted ComBat model, residual dataset, fitted regression model, etc.

## Diagram
![ComBatFamQC Diagram](/figure/ComBatFamQC_diagram.png)

## Package Features

The ComBatFamQC package offers the following five key functionalities:

1. <u>**Interactive Batch Effect Diagnostics & Harmonization**</u>

-   **Batch Effect Diagnostics**: ComBatFamQC provides two types of batch effect diagnostics methods for both individual batch effects and global batch effects: 1) Qualitative Visualization and 2) Statistical Testing. It simplifies the process of performing statistical analyses to detect potential batch effects and provides all relevant statistical test results for batch effect visualization and evaluation.

-   **Harmonization**: ComBatFamQC also provides four types of commonly used harmonization techniques, integrated through the [ComBatFamily](https://github.com/andy1764/ComBatFamily) package developed by Dr. Andrew Chen, for users to consider. The four harmonization techniques include: 
    -   Original ComBat
    -   Longitudinal ComBat
    -   ComBat-GAM
    -   CovBat.

-   **Interactive Visualization through R Shiny**: The ComBatFamQC package comes with an interactive visualization tool built on R Shiny, providing an intuitive user interface to explore and evaluate batch effects, as well as conduct interactive harmonization if needed. The output is organized into multiple tabs, which includes:

    -   **Data Overview**: Complete data overview and exploratory analysis
    -   **Summary**: Sample Size and Covariate Distribution
    -   **Residual Plot**: Additive and Multiplicative Batch Effect
    -   **Diagnosis of Global Batch Effect**: PCA, T-SNE and MDMR
    -   **Diagnosis of Individual Batch Effect**:
        -   *Statistical Tests for Additive Batch Effect*: Kenward-Roger (liner mix model), ANOVA, Kruskal-Wallis
        -   *Statistical Tests for Multiplicative Batch Effect*: Fligner-Killeen, Levene's Test, Bartlett's Test
    -   **Harmonization** Interactive Harmonization if needed

2. <u>**Post-Harmonization Downstream Analysis**</u>

-   **Age Trajectory** \
    Generate age trend of each brain structure (roi), adjusting sex and ICV. Customized centiles are enabled as well.
    -  **Age Trend Plots**
    -  **Age Trend Table** 

-   **Residual Generation** \
    Generate residual data set, removing specific covariates' effetcs.


## Installation

```{r}
library(devtools)

devtools::install_github("Zheng206/ComBatFam_Pipeline/ComBatFamQC", build_vignettes = TRUE)

```

## Tutorial

```{r}
vignette("ComBatQC")
vignette("Post-Harmonization")
```