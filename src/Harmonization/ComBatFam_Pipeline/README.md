# ComBatFamily Pipeline

## Summary

The ComBatFamily Pipeline includes **1)** ComBat Family harmonization methods, **2)** a visualization tool that provides interactive diagnostics and statistical analysis to evaluate how batch effects impact the data, and **3)** post-harmonization steps for ROIs' age trend visualization and residual generation, eliminating unwanted covariates' effects.

To simplify the harmonization process, ComBatFamily Pipeline seamlessly integrates both the harmonization methods and the visualization tool into a single, unified command line interface (**combatQC_CLI.R**). This integration enables users to execute both harmonization and visualization steps in a smooth and efficient manner. Beyond, the ComBatFamily Pipleline also includes post-harmonization steps to further investigate life span age trend of brain structures and other significant variables' effects on brain structures, removing unwanted covariates. Another unified command line interface (**post_CLI.R**) is designed for the post harmonization steps.

In a short summary, the following two command line interfaces are included in the ComBatFamily pipeline:

-   **Harmonization & Visualization**: combatQC_CLI.R
    -   Batch effect diagnostics and visualization
    -   Data Harmonization
-   **Post-Harmonization**: post_CLI.R
    -   Life span age trend of brain structures visualization
    -   residual data set removing unwanted covariates' effects

## Diagram

![ComBatFamily Pipeline Diagram](/figure/pipeline_diagram.png)

## Usage

Use the following command to access the Harmonization & Visualization pipeline:

```
Rscript combatQC_CLI.R --help
```

Use the following command to access the Harmonization & Visualization pipeline:

```
Rscript post_CLI.R --help
```

This step will display the available commands and options for both pipelines.

## Package Installation

-   ComBatFamily

```{r}
library(devtools)

devtools::install_github("Zheng206/ComBatFam_Pipeline/ComBatFamily", build_vignettes = TRUE)

```

-   ComBatFamQC

```{r}
library(devtools)

devtools::install_github("Zheng206/ComBatFam_Pipeline/ComBatFamQC", build_vignettes = TRUE)
```