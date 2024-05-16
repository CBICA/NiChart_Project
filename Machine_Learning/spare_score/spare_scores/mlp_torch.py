import logging
import time

import numpy as np
from spare_scores.data_prep import logging_basic_config

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, mean_squared_error, roc_auc_score, mean_absolute_error, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import optuna

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, num_features = 147, hidden_size = 256, classification = True, dropout = 0.2, use_bn = False, bn = 'bn'):
        super(SimpleMLP, self).__init__()

        self.num_features   = num_features
        self.hidden_size    = hidden_size 
        self.dropout        = dropout
        self.classification = classification
        self.use_bn         = use_bn

        self.linear1 = nn.Linear(self.num_features, self.hidden_size)
        self.norm1 = nn.InstanceNorm1d(self.hidden_size , eps=1e-15) if bn != 'bn' else nn.BatchNorm1d(self.hidden_size, eps=1e-15)

        self.linear2 = nn.Linear(self.hidden_size,  self.hidden_size//2)
        self.norm2 = nn.InstanceNorm1d(self.hidden_size //2 , eps=1e-15) if bn != 'bn' else nn.BatchNorm1d(self.hidden_size //2, eps=1e-15)

        self.linear3 = nn.Linear(self.hidden_size//2 , 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## first layer
        x = self.linear1(x)
        if self.use_bn:
            x = self.norm1(x)
        x = self.dropout(self.relu(x))

        ## second layer
        x = self.linear2(x)
        if self.use_bn:
            x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        ## thrid layer
        x = self.linear3(x)

        if self.classification:
            x = self.sigmoid(x)
        else:
            x = self.relu(x)

        return x.squeeze()

class MLPTorchModel:
    """
    A class for managing MLP models.

    Static attributes:
        predictors (list): List of predictors used for modeling.
        to_predict (str): Target variable for modeling.
        key_var (str): Key variable for modeling.

    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.

    Methods:
        train_model(df, **kwargs):
            Trains the model using the provided dataframe.
        
        apply_model(df):
            Applies the trained model on the provided dataframe and returns
            the predictions.
        
        set_parameters(**parameters):
            Updates the model's parameters with the provided values. This also
            changes the model's attributes, while retaining the original ones.
    """
    def __init__(self, predictors, to_predict, key_var, verbose=1,**kwargs):
        logger = logging_basic_config(verbose, content_only=True)
        
        self.predictors = predictors
        self.to_predict = to_predict
        self.key_var = key_var
        self.verbose = verbose

        valid_parameters = ['task', 'bs', 'num_epoches']

        for parameter in kwargs.keys():
            if parameter not in valid_parameters:
                print("Parameter '" + parameter + "' is not accepted for "
                        +"MLPModel. Ignoring...")
                continue
            
            if parameter == 'task':
                if kwargs[parameter] not in ['Classification', 'Regression']:
                    logger.error("Only 'Classification' and 'Regression' "
                                    + "tasks are supported.")
                    raise ValueError("Only 'Classification' and 'Regression' "
                                    + "tasks are supported.")
                else:
                    self.task = kwargs[parameter]
                continue

            if parameter == 'bs':
                try:
                    self.batch_size = int(kwargs[parameter])
                except ValueError:
                    print('Parameter: # of bs should be integer')

            if parameter == 'num_epoches':
                try:
                    self.num_epochs = int(kwargs[parameter])
                except ValueError:
                    print('Parameter: # of num_epoches should be integer')

            self.__dict__.update({parameter: kwargs[parameter]})

        # Set default values for the parameters if not provided

        if 'task' not in kwargs.keys():
            self.task = 'Regression'

        if 'batch_size' not in kwargs.keys():
            self.batch_size = 128

        if 'num_epochs' not in kwargs.keys():
            self.num_epochs = 100

        if device != 'cuda':
            print('You are not using the GPU! Check your device')

        ################################## MODEL SETTING ##################################################
        self.classification = True if self.task == 'Classification' else False
        self.mdl        = None
        self.scaler     = None
        self.stats      = None
        self.param      = None
        self.train_dl   = None
        self.val_dl     = None
        ################################## MODEL SETTING ##################################################

    def find_best_threshold(self, y_hat, y):
        fpr, tpr, thresholds_roc = roc_curve(y, y_hat, pos_label= 1)
        youden_index = tpr - fpr
        best_threshold_youden = thresholds_roc[np.argmax(youden_index)]

        return best_threshold_youden

    def get_all_stats(self, y_hat, y, classification = True):
        """
        Input: 
            y:     ground truth y (1: AD, 0: CN) -> numpy 
            y_hat: predicted y -> numpy, notice y_hat is predicted value [0.2, 0.8, 0.1 ...]

        Output:
            A dictionary contains: Acc, F1, Sensitivity, Specificity, Balanced Acc, Precision, Recall
        """
        y = np.array(y)
        y_hat = np.array(y_hat)

        if classification: 
            auc = roc_auc_score(y, y_hat) if len(set(y)) != 1 else 0.5

            self.threshold = self.find_best_threshold(y_hat, y)
 
            y_hat = np.where(y_hat >= self.threshold, 1 , 0)

            
            dict = {}
            dict['Accuracy']          = accuracy_score(y, y_hat)
            dict['AUC']               = auc
            dict['Sensitivity']       = 0
            dict['Specificity']       = 0
            dict['Balanced Accuarcy'] = balanced_accuracy_score(y, y_hat)
            dict['Precision']         = precision_score(y, y_hat)
            dict['Recall']            = recall_score(y, y_hat)
            dict['F1']                = f1_score(y, y_hat)

            if len(set(y)) != 1:
                tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                dict['Sensitivity']       = sensitivity
                dict['Specificity']       = specificity
        else:
            dict = {}
            mae  = mean_absolute_error(y, y_hat)
            mrse = mean_squared_error(y, y_hat, squared=False)
            r2   = r2_score(y, y_hat)
            dict['MAE']  = mae
            dict['RMSE'] = mrse
            dict['R2']   = r2

        
        return dict 
    

    def object(self, trial):

        evaluation_metric = 'Balanced Accuarcy' if self.task == 'Classification' else 'MAE'

        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
        dropout     = trial.suggest_float('dropout', 0.1, 0.8, step = 0.05)
        lr          = trial.suggest_float('lr', 1e-4, 1e-1, log = True)
        use_bn      = trial.suggest_categorical('use_bn', [False, True])
        bn          = trial.suggest_categorical('bn', ['bn', 'in'])

        model = SimpleMLP(num_features =len(self.predictors), 
                          hidden_size = hidden_size,
                          classification= self.classification, 
                          dropout= dropout, 
                          use_bn= use_bn, 
                          bn = bn).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = lr)
        loss_fn = nn.BCELoss() if self.classification else nn.L1Loss()

        model.train()

        for epoch in range(self.num_epochs):

            step = 0

            for _, (x,y) in enumerate(self.train_dl):
                step += 1
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                optimizer.zero_grad()

                loss = loss_fn(output, y)

                loss.backward()

                optimizer.step()


            val_step = 0
            val_total_metric = 0
            val_total_loss = 0

            with torch.no_grad():
                for _, (x, y) in enumerate(self.val_dl):
                    val_step += 1
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x.float())

                    loss = loss_fn(output, y)
                    val_total_loss += loss.item()
                    metric = self.get_all_stats(output.cpu().data.numpy(), y.cpu().data.numpy() , classification= self.classification)[evaluation_metric]
                    val_total_metric += metric

                val_total_loss = val_total_loss / val_step
                val_total_metric  = val_total_metric / val_step 

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # save checkpoint 
            trial.report(val_total_loss, epoch)
            checkpoint = {
                'trial_params': trial.params,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'validation_loss' : val_total_loss
            }
            trial.set_user_attr('checkpoint', checkpoint)
            
        return val_total_metric
    
   
    def set_parameters(self, **parameters):
        if 'linear1.weight' in parameters.keys():
            self.param = parameters
        else:
            self.__dict__.update(parameters)
        
    @ignore_warnings(category= (ConvergenceWarning,UserWarning))
    def fit(self, df, verbose=1, **kwargs):
        logger = logging_basic_config(verbose, content_only=True)
        
        
        # Time the training:
        start_time = time.time()

        logger.info(f'Training the MLP model...')
        
        ############################################ start training model here ####################################
        X = df[self.predictors]
        y = df[self.to_predict].tolist()

        stratify = y if self.task == 'Classification' else None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify= stratify)

        X_train = X_train.reset_index(drop = True)
        X_val = X_val.reset_index(drop = True)

        self.scaler = StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val  = self.scaler.transform(X_val)

        train_ds = MLPDataset(X_train, y_train)
        val_ds   = MLPDataset(X_val, y_val)

        self.train_dl = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)
        self.val_dl   = DataLoader(val_ds, batch_size = self.batch_size, shuffle = True)

        study = optuna.create_study(direction= 'maximize' if self.classification else 'minimize')
        study.optimize(self.object, n_trials= 100)

        # Get the best trial and its checkpoint
        best_trial = study.best_trial
        best_checkpoint = best_trial.user_attrs['checkpoint']

        best_hyperparams = best_checkpoint['trial_params']
        best_model_state_dict = best_checkpoint['model_state_dict']
        
        self.mdl = SimpleMLP(num_features = len(self.predictors), 
                             hidden_size = best_hyperparams['hidden_size'], 
                             classification= self.classification, 
                             dropout= best_hyperparams['dropout'], 
                             use_bn= best_hyperparams['use_bn'], 
                             bn = best_hyperparams['bn'])

        self.mdl.load_state_dict(best_model_state_dict)
        self.mdl.to(device)
        self.mdl.eval()
        X_total = self.scaler.transform( np.array(X, dtype = np.float32) )
        X_total = torch.tensor(X_total).to(device)
        
        self.y_pred = self.mdl(X_total).cpu().data.numpy()
        self.stats = self.get_all_stats(self.y_pred, y, classification = self.classification)

        self.param =  best_model_state_dict

        ########################################################################################################### 

        training_time = time.time() - start_time
        self.stats['training_time'] = round(training_time, 4)


        result = {'predicted':self.y_pred, 
                  'model':self.mdl, 
                  'stats':self.stats, 
                  'best_params': self.param,
                  'CV_folds': None,
                  'scaler': self.scaler}
    
        if self.task == 'Regression':
            print('>>MAE = ', self.stats['MAE'])
            print('>>RMSE = ', self.stats['RMSE'])
            print('>>R2 = ', self.stats['R2'])

        else:
            print('>>AUC = ', self.stats['AUC'])
            print('>>Accuracy = ', self.stats['Accuracy'])
            print('>>Sensityvity = ', self.stats['Sensitivity'])
            print('>>Specificity = ', self.stats['Specificity'])
            print('>>Precision = ', self.stats['Precision'])
            print('>>Recall = ', self.stats['Recall'])
            print('>>F1 = ', self.stats['F1'])
            print('>>Threshold = ', self.threshold)

        return result 
    
    def predict(self, df):
        X = df[self.predictors]
        X = self.scaler.transform(np.array(X, dtype = np.float32))
        X = torch.tensor(X).to(device)

        checkpoint_dict = self.param
        self.mdl.load_state_dict(checkpoint_dict)
        self.mdl.eval()

        y_pred = self.mdl(X).cpu().data.numpy()

        return y_pred

    def output_stats(self):
        [logging.info(f'>> {key} = {np.mean(value):#.4f} \u00B1 {np.std(value):#.4f}') for key, value in self.stats.items()]
