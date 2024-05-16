************
QuickStart
************

After installation, users can quickly run a small example in Python:

**1. Load example data and model**

   .. code-block:: python

      import pandas as pd
      import spare_scores as spare
      import spare_scores.data_prep as data_prep

      df = data_prep.load_examples('example_data.csv')
      mdl, meta_data = data_prep.load_examples(f'mdl_SPARE_BA_hMUSE_single.pkl.gz')

**2. Test the model on the example data**

   .. code-block:: python
      
      df['SPARE_BA'] = spare.spare_test(df, (mdl, meta_data))

**3. Train another regression model**

   .. code-block:: python
      
      predictors = df.columns[df.columns.str.startswith('H_MUSE')][:20]
      mdl_1, meta_data_1 = spare.spare_train(df, 'Age', data_vars=predictors)
      mdl_2, meta_data_1 = spare.spare_train(df, 'Sex', data_vars=predictors, pos_group='M')