# DKGP: Deep Kernel Gaussian Process models

Pipeline name: DKGP


Pipeline description:
    Deep Kernel Gaussian Process models for population-level temporal data analysis, specifically designed for medical imaging applications and biomarker trajectory prediction.
    The DKGP model combines deep neural networks with Gaussian Processes to provide:
        Population-level modeling of temporal trajectories
        Uncertainty quantification with confidence intervals that correspond to the 95% percentile of the posterior predictive distribution
        Deep feature learning for complex temporal patterns
        Production-ready inference for new subjects
        8-year trajectory forecasting with 12-month intervals

Input:
    - T1 MRI scans
    - CSV columns: MRID, Age, Sex, AD_Diagnosis ('CN', 'MCI', or 'AD'), ADAS_COG_13, MMSE
