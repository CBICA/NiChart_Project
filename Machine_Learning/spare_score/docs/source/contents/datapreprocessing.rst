*******************
Data Pre-Processing
*******************

Users can bring their own dataset to train SPARE models. Here are steps to prepare your dataset for the training:

**1. Load your dataset as a Pandas DataFrame**

   .. code-block:: python

      import pandas as pd
      
      df = pd.read_csv('your_dataset.csv', low_memory=False)

**2. Define a column to predict and columns to use as predictors**

   .. code-block:: python

      to_predict = 'binary_variable' # for a SPARE classification model
      to_predict = 'continuous_variable" # for a SPARE regression model

      predictors = df.columns.str.startswith('MUSE_Volume_')
      # categorical variables with more than 2 categories are currently not supported.

**3. (Optional) Select unique timepoints for a longitudinal dataset**

   .. code-block:: python

      import spare_scores.data_prep as data_prep

      df = data_prep.smart_unique(df, to_predict=to_predict)
      # selects unique timepoints in a way that optimizes SPARE training.

**4. (Optional) Match two groups to classify for age and sex**

   .. code-block:: python

      df = data_prep.age_sex_match(df, to_match=to_predict, p_threshold=0.15)
      # matches two groups for age and sex in a way that optimizes SPARE training.