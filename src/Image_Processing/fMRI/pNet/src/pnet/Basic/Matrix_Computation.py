# Yuncong Ma, 2/2/2024
# basic matrix computation functions

#########################################
# Packages
import torch
import numpy as np
# disable grad computation on 08/03/2024
torch.set_grad_enabled(False)
# other functions of pNet
from Module.Data_Input import set_data_precision, set_data_precision_torch


def mat_corr_(X, Y=None, dataPrecision='double'):
    """
    mat_corr(X, Y=None, dataPrecision='double')
    Perform corr as in MATLAB, pair-wise Pearson correlation between columns in X and Y

    :param X: 1D or 2D matrix
    :param Y: 1D or 2D matrix, or None
    :param dataPrecision: 'double' or 'single'
    X and Y have the same number of rows
    :return: Corr

    Note: this method will use memory as it concatenates X and Y along column direction.
    By Yuncong Ma, 9/5/2023
    """

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np_float)
    else:
        X = X.astype(np_float)
    if Y is not None:
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y, dtype=np_float)
        else:
            Y = Y.astype(np_float)

    # Check size of X and Y
    if len(X.shape) > 2 or (Y is not None and len(Y.shape) > 2):
        raise ValueError("X and Y must be 1D or 2D matrices")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of columns")

    dim_X = X.shape
    if Y is not None:
        dim_Y = Y.shape
        if len(dim_X) == 2 and len(dim_Y) == 2:
            temp = np.corrcoef(X, Y, rowvar=False)
            Corr = temp[0:dim_X[1], dim_X[1]:dim_X[1]+dim_Y[1]]
        elif len(dim_X) == 1 and len(dim_Y) == 2:
            temp = np.corrcoef(X, Y, rowvar=False)
            Corr = temp[0, 1:1+dim_Y[1]]
        elif len(dim_X) == 2 and len(dim_Y) == 1:
            temp = np.corrcoef(X, Y, rowvar=False)
            Corr = temp[1:1+dim_X[1], 0]
        else:
            temp = np.corrcoef(X, Y, rowvar=False)
            Corr = temp[0, 1]
    else:
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D matrix")
        Corr = np.corrcoef(X, rowvar=False)

    return Corr


def mat_corr(X, Y=None, dataPrecision='double'):
    """
    Perform corr as in MATLAB, pair-wise Pearson correlation between columns in X and Y

    :param X: 1D or 2D matrix, numpy.ndarray or torch.Tensor
    :param Y: 1D or 2D matrix, or None, numpy.ndarray or torch.Tensor
    :param dataPrecision: 'double' or 'single'
    X and Y have the same number of rows
    :return: Corr

    Note: this method will use memory as it concatenates X and Y along column direction.
    #modified version of the torch corr  on 08/05/2024

    """

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(X, np.ndarray):
        X = np.ndarray(X, dtype=np_float)
    else:
        X = X.astype(np_float)
    if Y is not None:
        if not isinstance(Y, np.ndarray):
            Y = np.ndarray(Y, dtype=np_float)
        else:
            Y = Y.astype(np_float)

    # Check size of X and Y
    if len(X.shape) > 2 or (Y is not None and len(Y.shape) > 2):
        raise ValueError("X and Y must be 1D or 2D matrices")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of columns")

    if Y is not None:
        # Subtract the mean to calculate the covariance
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        # Compute the standard deviation of the columns
        std_X = np.std(X_centered, axis=0, keepdims=True, ddof=1)
        std_Y = np.std(Y_centered, axis=0, keepdims=True, ddof=1)
        # Compute the correlation matrix
        numerator = (X_centered.T @ Y_centered)
        denominator = (std_X.T @ std_Y) * np.array(X.shape[0] - 1)
        Corr = numerator / (denominator + np_eps)
    else:
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D matrix")
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        std_X = np.std(X_centered, axis=0, keepdims=True, ddof=1)
        numerator = (X_centered.T @ X_centered)
        denominator = (std_X.T @ std_X) * np.array(X.shape[0] - 1)
        Corr = numerator / (denominator + np.eps)

    return Corr


def mat_corr_torch(X, Y=None, dataPrecision='double'):
    """
    Perform corr as in MATLAB, pair-wise Pearson correlation between columns in X and Y

    :param X: 1D or 2D matrix, numpy.ndarray or torch.Tensor
    :param Y: 1D or 2D matrix, or None, numpy.ndarray or torch.Tensor
    :param dataPrecision: 'double' or 'single'
    X and Y have the same number of rows
    :return: Corr

    Note: this method will use memory as it concatenates X and Y along column direction.
    By Yuncong Ma, 12/6/2023
    """

    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch_float)
    else:
        X = X.type(torch_float)
    if Y is not None:
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch_float)
        else:
            Y = Y.type(torch_float)

    # Check size of X and Y
    if len(X.shape) > 2 or (Y is not None and len(Y.shape) > 2):
        raise ValueError("X and Y must be 1D or 2D matrices")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of columns")

    if Y is not None:
        # Subtract the mean to calculate the covariance
        X_centered = X - torch.mean(X, dim=0, keepdim=True)
        Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)
        # Compute the standard deviation of the columns
        std_X = torch.std(X_centered, dim=0, keepdim=True, unbiased=True)
        std_Y = torch.std(Y_centered, dim=0, keepdim=True, unbiased=True)
        # Compute the correlation matrix
        numerator = (X_centered.T @ Y_centered)
        denominator = (std_X.T @ std_Y) * torch.tensor(X.shape[0] - 1)
        Corr = numerator / denominator
    else:
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D matrix")
        X_centered = X - torch.mean(X, dim=0, keepdim=True)
        std_X = torch.std(X_centered, dim=0, keepdim=True, unbiased=True)
        numerator = (X_centered.T @ X_centered)
        denominator = (std_X.T @ std_X) * torch.tensor(X.shape[0] - 1)
        Corr = numerator / denominator

    return Corr