# Yuncong Ma, 1/10/2024
# FN Computation module of pNet

#########################################
# Packages
import numpy
import numpy as np
import scipy
import scipy.io as sio
import os
import re
import time

# other functions of pNet
from Module.Data_Input import *


def mat_corr(X, Y=None, dataPrecision='double'):
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


def normalize_data(data, algorithm='vp', normalization='vmax', dataPrecision='double'):
    """
    normalize_data(data, algorithm='vp', normalization='vmax', dataPrecision='double')
    Normalize data by algorithm and normalization settings

    :param data: data in 2D matrix [dim_time, dim_space], recommend to use reference mode to save memory
    :param algorithm: 'z' 'gp' 'vp'
    :param normalization: 'n2' 'n1' 'rn1' 'g' 'vmax'
    :param dataPrecision: 'double' or 'single'
    :return: data

    Consistent to MATLAB function normalize_data(X, algorithm, normalization, dataPrecision)
    By Yuncong Ma, 9/8/2023
    """

    if len(data.shape) != 2:
        raise ValueError("Data must be a 2D matrix")

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np_float)
    else:
        data = data.astype(np_float)

    if algorithm.lower() == 'z':
        # standard score for each variable
        mVec = np.mean(data, axis=1)
        sVec = np.maximum(np.std(data, axis=1), np_eps)
        data = (data - mVec[:, np.newaxis]) / sVec[:, np.newaxis]
    elif algorithm.lower() == 'gp':
        # remove negative value globally
        minVal = np.min(data)
        shiftVal = np.abs(np.minimum(minVal, 0))
        data = data + shiftVal
    elif algorithm.lower() == 'vp':
        # remove negative value voxel-wisely
        minVal = np.min(data, axis=0, keepdims=True)
        shiftVal = np.abs(np.minimum(minVal, 0))
        data += shiftVal
    else:
        # do nothing
        data = data

    if normalization.lower() == 'n2':
        # l2 normalization for each observation
        l2norm = np.sqrt(np.sum(data ** 2, axis=1)) + np_eps
        data = data / l2norm[:, np.newaxis]
    elif normalization.lower() == 'n1':
        # l1 normalization for each observation
        l1norm = np.sum(data, axis=1) + np_eps
        data = data / l1norm[:, np.newaxis]
    elif normalization.lower() == 'rn1':
        # l1 normalization for each variable
        l1norm = np.sum(data, axis=0) + np_eps
        data = data / l1norm
    elif normalization.lower() == 'g':
        # global scale
        sVal = np.sort(data, axis=None)
        perT = 0.001
        minVal = sVal[int(len(sVal) * perT)]
        maxVal = sVal[int(len(sVal) * (1 - perT))]
        data[data < minVal] = minVal
        data[data > maxVal] = maxVal
        data = (data - minVal) / max((maxVal - minVal), np_eps)
    elif normalization.lower() == 'vmax':
        cmin = np.min(data, axis=0, keepdims=True)
        cmax = np.max(data, axis=0, keepdims=True)
        data = (data - cmin) / np.maximum(cmax - cmin, np_eps)
    else:
        # do nothing
        data = data

    if np.isnan(data).any():
        raise ValueError('  nan exists, check the preprocessed data')

    return data


def initialize_u(X, U0, V0, error=1e-4, maxIter=1000, minIter=30, meanFitRatio=0.1, initConv=1, dataPrecision='double'):
    """
    initialize_u(X, U0, V0, error=1e-4, maxIter=1000, minIter=30, meanFitRatio=0.1, initConv=1, dataPrecision='double')

    :param X: data, 2D matrix [dim_time, dim_space]
    :param U0: initial temporal component, 2D matrix [dim_time, k]
    :param V0: initial spatial component, 2D matrix [dim_space, k]
    :param error: data fitting error
    :param maxIter: maximum iteration
    :param minIter: minimum iteration
    :param meanFitRatio: a 0-1 scaler, exponential moving average coefficient
    :param initConv: 0 or 1, flag for convergence
    :param dataPrecision: 'double' or 'single'
    :return: U_final: temporal components of FNs, a 2D matrix [dim_time, K]

    Consistent to MATLAB function initialize_u(X, U0, V0, error, maxIter, minIter, meanFitRatio, initConv, dataPrecision)
    By Yuncong Ma, 9/5/2023
    """

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np_float)
    else:
        X = X.astype(np_float)
    if not isinstance(U0, np.ndarray):
        U0 = np.array(U0, dtype=np_float)
    else:
        U0 = U0.astype(np_float)
    if not isinstance(V0, np.ndarray):
        V0 = np.array(V0, dtype=np_float)
    else:
        V0 = V0.astype(np_float)

    # Check the data size of X, U0 and V0
    if len(X.shape) != 2 or len(U0.shape) != 2 or len(V0.shape) != 2:
        raise ValueError("X, U0 and V0 must be 2D matrices")
    if X.shape[0] != U0.shape[0] or X.shape[1] != V0.shape[0] or U0.shape[1] != V0.shape[1]:
        raise ValueError("X, U0 and V0 need to have appropriate sizes")

    # Duplicate a copy for iterative update of U and V
    U = U0.copy()
    V = V0.copy()

    newFit = data_fitting_error(X, U, V, 0, 1, dataPrecision)
    meanFit = newFit / meanFitRatio

    maxErr = 1
    for i in range(1, maxIter+1):
        # update U with fixed V
        XV = X @ V
        VV = V.T @ V
        UVV = U @ VV

        U = U * (XV / np.maximum(UVV, np_eps))

        if i > minIter:
            if initConv:
                newFit = data_fitting_error(X, U, V, 0, 1, dataPrecision)
                meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newFit
                maxErr = (meanFit - newFit) / meanFit

        if maxErr <= error:
            break

    U_final = U
    return U_final


def data_fitting_error(X, U, V, deltaVU=0, dVordU=1, dataPrecision='double'):
    """
    data_fitting_error(X, U, V, deltaVU, dVordU, dataPrecision='double')
    Calculate the datat fitting of X'=UV' with terms

    :param X: 2D matrix, [Space, Time]
    :param U: 2D matrix, [Time, k]
    :param V: 2D matrix, [Space, k]
    :param deltaVU: 0
    :param dVordU: 1
    :param dataPrecision: 'double' or 'single'
    :return: Fitting_Error

    Consistent to MATLAB function fitting_initialize_u(X, U, V, deltaVU, dVordU, dataPrecision)
    By Yuncong Ma, 9/6/2023
    """

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np_float)
    else:
        X = X.astype(np_float)
    if not isinstance(U, np.ndarray):
        U = np.array(U, dtype=np_float)
    else:
        U = U.astype(np_float)
    if not isinstance(V, np.ndarray):
        V = np.array(V, dtype=np_float)
    else:
        V = V.astype(np_float)

    # Check data size of X, U and V
    if len(X.shape) != 2 or len(U.shape) != 2 or len(V.shape) != 2:
        raise ValueError("X, U and V must be 2D matrices")
    if X.shape[0] != U.shape[0] or X.shape[1] != V.shape[0] or U.shape[1] != V.shape[1]:
        raise ValueError("X, U and V need to have appropriate sizes")

    dV = []
    maxM = 62500000  # To save memory
    dim_time, dim_space = X.shape
    mn = np.prod(X.shape)
    nBlock = int(np.floor(mn*3/maxM))

    if mn < maxM:
        dX = U @ V.T - X
        obj_NMF = np.sum(np.power(dX, 2))
        if deltaVU:
            if dVordU:
                dV = dX.T * U
            else:
                dV = dX * V
    else:
        obj_NMF = 0
        if deltaVU:
            if dVordU:
                dV = np.zeros_like(V)
            else:
                dV = np.zeros_like(U)
        for i in range(int(np.ceil(dim_space/nBlock))):
            if i == int(np.ceil(dim_space/nBlock)):
                smpIdx = range(i*nBlock, dim_space)
            else:
                smpIdx = range(i*nBlock, np.minimum(dim_space, (i+1)*nBlock))
            dX = (U @ V[smpIdx, :].T) - X[:, smpIdx]
            obj_NMF += np.sum(np.power(dX, 2))
            if deltaVU:
                if dVordU:
                    dV[smpIdx, :] = np.dot(dX.T, U)
                else:
                    dV += np.dot(dX, V[smpIdx, :])
        if deltaVU:
            if dVordU:
                dV = dV

    Fitting_Error = obj_NMF
    return Fitting_Error


def normalize_u_v(U, V, NormV, Norm, dataPrecision='double'):
    """
    normalize_u_v(U, V, NormV, Norm, dataPrecision='double')
    Normalize U and V with terms

    :param U: 2D matrix, [Time, k]
    :param V: 2D matrix, [Space, k]
    :param NormV: 1 or 0
    :param Norm: 1 or 2
    :param dataPrecision: 'double' or 'single'
    :return: U, V

    Consistent to MATLAB function normalize_u_v(U, V, NormV, Norm, dataPrecision)
    By Yuncong Ma, 9/5/2023
    """

    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(U, np.ndarray):
        U = np.array(U, dype=np_float)
    else:
        U = U.astype(np_float)
    if not isinstance(V, np.ndarray):
        V = np.array(V, dype=np_float)
    else:
        V = V.astype(np_float)

    # Check data size of U and V
    if len(U.shape) != 2 or len(V.shape) != 2:
        raise ValueError("U and V must be 2D matrices")
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V need to have appropriate sizes")

    dim_space = V.shape[0]
    dim_time = U.shape[0]

    if Norm == 2:
        norms = np.sqrt(np.sum(np.power(V, 2), axis=0))
        norms = np.maximum(norms, np_eps)
    else:
        norms = np.max(V, axis=0)
        norms = np.maximum(norms, np_eps)

    if NormV:
        U = U * np.tile(norms, (dim_time, 1))
        V = V / np.tile(norms, (dim_space, 1))
    else:
        U = U / np.tile(norms, (dim_time, 1))
        V = V * np.tile(norms, (dim_space, 1))

    return U, V


def construct_Laplacian_gNb(gNb: np.ndarray, dim_space, vxI=0, X=None, alphaL=10, normW=1, dataPrecision='double'):
    """
    construct_Laplacian_gNb(gNb: np.ndarray, dim_space, vxI=0, X=None, alphaL=10, normW=1, dataPrecision='double')
    construct Laplacian matrices for Laplacian spatial regularization term

    :param gNb: graph neighborhood, a 2D matrix [N, 2] storing rows and columns of non-zero elements
    :param dim_space: dimension of space (number of voxels or vertices)
    :param vxI: 0 or 1, flag for using the temporal correlation between nodes (vertex, voxel)
    :param X: fMRI data, a 2D matrix, [dim_time, dim_space]
    :param alphaL: internal hyper parameter for Laplacian regularization term
    :param normW: 1 or 2, normalization method for Laplacian matrix W
    :param dataPrecision: 'double' or 'single'
    :return: L, W, D: sparse 2D matrices [dim_space, dim_space]

    Yuncong Ma, 9/13/2023
    """

    np_float, np_eps = set_data_precision(dataPrecision)

    # Construct the spatial affinity graph
    # gNb uses index starting from 1
    W = scipy.sparse.coo_matrix((np.ones(gNb.shape[0]), (gNb[:, 0] - 1, gNb[:, 1] - 1)), shape=(dim_space, dim_space), dtype=np_float)
    if vxI > 0:
        for i in range(gNb.shape[0]):
            xi = gNb[i, 0] - 1
            yi = gNb[i, 1] - 1
            if xi < yi:
                corrVal = (1.0 + mat_corr(X[:, xi], X[:, yi], dataPrecision)) / 2
                W[xi, yi] = corrVal
                W[yi, xi] = corrVal

    # Defining other matrices
    DCol = np.array(W.sum(axis=1), dtype=np_float).flatten()
    D = scipy.sparse.spdiags(DCol, 0, dim_space, dim_space)
    L = D - W
    if normW > 0:
        D_mhalf = scipy.sparse.spdiags(np.power(DCol, -0.5), 0, dim_space, dim_space)
        L = D_mhalf @ L @ D_mhalf * alphaL
        W = D_mhalf @ W @ D_mhalf * alphaL
        D = D_mhalf @ D @ D_mhalf * alphaL

    return L, W, D

def robust_normalize_V(V, factor = 0.95, dataPrecision='double'):
    # Setup data precision and eps
    np_float, np_eps = set_data_precision(dataPrecision)
    # assume V is the FNs with its first dim as spatial one
    #robust normalization of V
    sdim, fdim = V.shape
    if sdim < fdim:
        print('\n  the input V has wrong dimension.' + str(sdim) + str(fdim) + '\n')
    vtop = np.percentile(V, 1.0-factor, axis=0, keepdims=True)
    vbottome = np.percentile(V, factor, axis=0, keepdims=True)
    vdiff = np.maximum(vtop-vbottome, np_eps)
    #print(vdiff)
    #print(torch.tile(vdiff, (V.shape[0], 1)).shape)
    V = np.clip( (V - np.tile(vbottome, (V.shape[0],1))) / np.tile(vdiff, (V.shape[0], 1)), 0.0, 1.0)
    return V

def pFN_NMF(Data, gFN, gNb, maxIter=1000, minIter=30, meanFitRatio=0.1, error=1e-4, normW=1,
            Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, initConv=1, ard=0, eta=0, dataPrecision='double', logFile='Log_pFN_NMF.log'):
    """
    pFN_NMF(Data, gFN, gNb, maxIter=1000, minIter=30,
            meanFitRatio=0.1, error=1e-4, normW=1,
            Alpha=2, Beta=30, alphaS=2, alphaL=10, initConv=1, ard=0, eta=0,
            dataPrecision='double', logFile='Log_pFN_NMF.log')
    Compute personalized FNs by spatially-regularized NMF method with group FNs as initialization

    :param Data: 2D matrix [dim_time, dim_space]. Data will be formatted to Tensor and normalized.
    :param gFN: group level FNs 2D matrix [dim_space, K], K is the number of functional networks. gFN will be cloned
    :param gNb: graph neighborhood, a 2D matrix [N, 2] storing rows and columns of non-zero elements
    :param maxIter: maximum iteration number for multiplicative update
    :param minIter: minimum iteration in case fast convergence
    :param meanFitRatio: a 0-1 scaler, exponential moving average coefficient, used for the initialization of U when using group initialized V
    :param error: difference of cost function for convergence
    :param normW: 1 or 2, normalization method for W used in Laplacian regularization
    :param Alpha: hyper parameter for spatial sparsity
    :param Beta: hyper parameter for Laplacian sparsity
    :param alphaS: internally determined, the coefficient for spatial sparsity based Alpha, data size, K, and gNb
    :param alphaL: internally determined, the coefficient for Laplacian sparsity based Beta, data size, K, and gNb
    :param vxI: 0 or 1, flag for using the temporal correlation between nodes (vertex, voxel)
    :param initConv: flag for convergence of initialization of U
    :param ard: 0 or 1, flat for combining similar clusters
    :param eta: a hyper parameter for the ard regularization term
    :param dataPrecision: 'single' or 'float32', 'double' or 'float64'
    :param logFile: str, directory of a txt log file
    :return: U and V. U is the temporal components of pFNs, a 2D matrix [dim_time, K], and V is the spatial components of pFNs, a 2D matrix [dim_space, K]

    Yuncong Ma, 12/13/2023
    """

    # Setup data precision and eps
    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(gFN, np.ndarray):
        gFN = np.array(gFN, dype=np_float)
    else:
        gFN = gFN.astype(np_float)
    if not isinstance(Data, np.ndarray):
        Data = np.array(Data, dype=np_float)
    else:
        Data = Data.astype(np_float)

    # check dimension of Data and gFN
    if Data.shape[1] != gFN.shape[0]:
        raise ValueError("The second dimension of Data should match the first dimension of gFn, as they are space dimension")

    K = gFN.shape[1]

    # setup log file
    logFile = open(logFile, 'a')
    print(f'\nStart NMF for pFN using NumPy at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # initialization
    initV = gFN.copy()

    dim_time, dim_space = Data.shape

    # Median number of graph neighbors
    nM = np.median(np.unique(gNb[:, 0], return_counts=True)[1])

    # Use Alpha and Beta to set alphaS and alphaL if they are 0
    if alphaS == 0 and Alpha > 0:
        alphaS = np.round(Alpha * dim_time / K)
    if alphaL == 0 and Beta > 0:
        alphaL = np.round(Beta * dim_time / K / nM)

    # Prepare and normalize scan
    Data = normalize_data(Data, 'vp', 'vmax', dataPrecision=dataPrecision)
    X = Data    # Save memory

    # Construct the spatial affinity graph
    L, W, D = construct_Laplacian_gNb(gNb, dim_space, vxI, X, alphaL, normW, dataPrecision)

    # Initialize V
    V = np.copy(initV)
    miv = np.max(V, axis=0)
    trimInd = V / np.maximum(np.tile(miv, (dim_space, 1)), np_eps) < 5e-2
    V[trimInd] = 0
    
    # robust normalization of V  added on 08/03/2024
    #V = robust_normalize_V(V, factor=1.0 / K)  # try to keep 1/K non-zero values for each component, updated on 08/03/2024

    # Initialize U
    U = X @ V / np.tile(np.sum(V, axis=0), (dim_time, 1))

    U = initialize_u(X, U, V, error=error, maxIter=100, minIter=30, meanFitRatio=meanFitRatio, initConv=initConv, dataPrecision=dataPrecision)

    initV = V.copy()

    # Alternative update of U and V
    # Variables

    if ard > 0:
        lambdas = np.sum(U, axis=0) / dim_time
        hyperLam = eta * np.sum(np.power(X, 2)) / (dim_time * dim_space * 2)
    else:
        lambdas = 0
        hyperLam = 0

    flagQC = 0
    oldLogL = np.inf
    oldU = U.copy()
    oldV = V.copy()
    #  Multiplicative update of U and V
    for i in range(1, 1+maxIter):
        # ===================== update V ========================
        # Eq. 8-11
        XU = X.T @ U
        UU = U.T @ U
        VUU = V @ UU

        tmpl2 = np.power(V, 2)

        if alphaS > 0:
            tmpl21 = np.sqrt(tmpl2)
            tmpl22 = np.tile(np.sqrt(np.sum(tmpl2, axis=0)), (dim_space, 1))
            tmpl21s = np.tile(np.sum(tmpl21, axis=0), (dim_space, 1))
            posTerm = V / np.maximum(tmpl21 * tmpl22, np_eps)
            negTerm = V * tmpl21s / np.maximum(np.power(tmpl22, 3), np_eps)

            VUU = VUU + 0.5 * alphaS * posTerm
            XU = XU + 0.5 * alphaS * negTerm

        if alphaL > 0:
            WV = W @ V.astype(np.float64)
            DV = D @ V.astype(np.float64)

            XU = XU + WV
            VUU = VUU + DV

        V = V * (XU / np.maximum(VUU, np_eps))

        # Prune V if empty components are found in V
        # This is almost impossible to happen without combining FNs
        prunInd = np.sum(V != 0, axis=0) == 1
        if np.any(prunInd):
            V[:, prunInd] = np.zeros((dim_space, np.sum(prunInd)), dype=np_float)
            U[:, prunInd] = np.zeros((dim_time, np.sum(prunInd)), dype=np_float)

        # normalize U and V
        U, V = normalize_u_v(U, V, 1, 1, dataPrecision=dataPrecision)

        # ===================== update U =========================
        XV = X @ V
        VV = V.T @ V
        UVV = U @ VV

        if ard > 0:  # ard term for U
            posTerm = 1 / np.maximum(np.tile(lambdas, (dim_time, 1)), np_eps)

            UVV = UVV + posTerm * hyperLam

        U = U * (XV / np.maximum(UVV, np_eps))

        # Prune U if empty components are found in U
        # This is almost impossible to happen without combining FNs
        prunInd = np.sum(U, axis=0) == 0
        if np.any(prunInd):
            V[:, prunInd] = np.zeros((dim_space, np.sum(prunInd)), dype=np_float)
            U[:, prunInd] = np.zeros((dim_time, np.sum(prunInd)), dype=np_float)

        # update lambda
        if ard > 0:
            lambdas = np.sum(U, axis=0) / dim_time

        # ==== calculate objective function value ====
        ardU = 0
        tmp2 = 0
        tmpl21 = np.power(V, 2)

        if ard > 0:
            su = np.sum(U, axis=0)
            su[su == 0] = 1
            ardU = np.sum(np.log(su)) * dim_time * hyperLam

        if alphaL > 0:
            tmp2 = V.T @ L * V.T

        L21 = alphaS * np.sum(np.sum(np.sqrt(tmpl21), axis=0) / np.maximum(np.sqrt(np.sum(tmpl21, axis=0)), np_eps))
        # LDf = data_fitting_error(X, U, V, 0, 1)
        LDf = np.sum(np.power(X - U @ V.T, 2))
        LSl = np.sum(tmp2)

        # Objective function
        LogL = L21 + ardU + LDf + LSl
        print(f"    Iter = {i}: LogL: {LogL}, dataFit: {LDf}, spaLap: {LSl}, L21: {L21}, ardU: {ardU}", file=logFile)

        # The iteration needs to meet minimum iteration number and small changes of LogL
        if i > minIter and abs(oldLogL - LogL) / np.maximum(oldLogL, np_eps) < error:
            break
        oldLogL = LogL.copy()

        # QC Control
        temp = mat_corr(gFN, V, dataPrecision)
        QC_Spatial_Correspondence = np.copy(np.diag(temp))
        temp -= np.diag(2 * np.ones(K))  # set diagonal values to lower than -1
        QC_Spatial_Correspondence_Control = np.max(temp, axis=1)
        QC_Delta_Sim = np.min(QC_Spatial_Correspondence - QC_Spatial_Correspondence_Control)

        if QC_Delta_Sim <= 0:
            flagQC = 1
            U = oldU.copy()
            V = oldV.copy()
            print(f'\n  QC: Meet QC constraint: Delta sim = {QC_Delta_Sim}', file=logFile, flush=True)
            print(f'    Use results from last iteration', file=logFile, flush=True)
            break
        else:
            oldU = U.copy()
            oldV = V.copy()
            print(f'        QC: Delta sim = {QC_Delta_Sim}', file=logFile, flush=True)

    print(f'\n Finished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    return U, V


def gFN_NMF(Data, K, gNb, init='random', maxIter=1000, minIter=200, error=1e-8, normW=1,
            Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, ard=0, eta=0, nRepeat=5, dataPrecision='double', logFile='Log_pFN_NMF.log'):
    """
    gFN_NMF(Data, K, gNb, maxIter=1000, minIter=30, error=1e-8, normW=1,
            Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, ard=0, eta=0, nRepeat=5, dataPrecision='double', logFile='Log_pFN_NMF.log')
    Compute group-level FNs using NMF method

    :param Data: 2D matrix [dim_time, dim_space], recommend to normalize each fMRI scan before concatenate them along the time dimension
    :param K: number of FNs
    :param gNb: graph neighborhood, a 2D matrix [N, 2] storing rows and columns of non-zero elements
    :param init: 'nndsvda': NNDSVD with zeros filled with the average of X (better when sparsity is not desired)  #updated on 08/03/2024
                 'random': non-negative random matrices, scaled with: sqrt(X.mean() / n_components)
                 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness)
                 'nndsvdar' NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired)
    :param maxIter: maximum iteration number for multiplicative update
    :param minIter: minimum iteration in case fast convergence
    :param error: difference of cost function for convergence
    :param normW: 1 or 2, normalization method for W used in Laplacian regularization
    :param Alpha: hyper parameter for spatial sparsity
    :param Beta: hyper parameter for Laplacian sparsity
    :param alphaS: internally determined, the coefficient for spatial sparsity based Alpha, data size, K, and gNb
    :param alphaL: internally determined, the coefficient for Laplacian sparsity based Beta, data size, K, and gNb
    :param vxI: flag for using the temporal correlation between nodes (vertex, voxel)
    :param ard: 0 or 1, flat for combining similar clusters
    :param eta: a hyper parameter for the ard regularization term
    :param nRepeat: Any positive integer, the number of repetition to avoid poor initialization
    :param dataPrecision: 'single' or 'float32', 'double' or 'float64'
    :param logFile: str, directory of a txt log file
    :return: gFN, 2D matrix [dim_space, K]

    Yuncong Ma, 11/27/2023
    """

    # setup log file
    if isinstance(logFile, str):
        logFile = open(logFile, 'a')
    print(f'\nStart NMF for gFN using NumPy at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # Setup data precision and eps
    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(Data, np.ndarray):
        Data = np.array(Data, dype=np_float)
    else:
        Data = Data.astype(np_float)

    # Input data size
    dim_time, dim_space = Data.shape

    # Median number of graph neighbors
    nM = np.median(np.unique(gNb[:, 0], return_counts=True)[1])

    # Use Alpha and Beta to set alphaS and alphaL if they are 0
    if alphaS == 0 and Alpha > 0:
        alphaS = np.round(Alpha * dim_time / K)
    if alphaL == 0 and Beta > 0:
        alphaL = np.round(Beta * dim_time / K / nM)

    # Prepare and normalize scan
    Data = normalize_data(Data, 'vp', 'vmax', dataPrecision)
    X = Data  # Save memory

    # Construct the spatial affinity graph
    L, W, D = construct_Laplacian_gNb(gNb, dim_space, vxI, X, alphaL, normW, dataPrecision)

    #add initialization parameter on 08/03/2024
    # to use sklearn NMF
    # from sklearn.decomposition import NMF   # for sklearn based NMF initialization
    # model = NMF(n_components=K, init=init, max_iter=1, solver='mu')
    # model._check_params(X)
    # U = None
    # V = None
    # U, V = model._check_w_h(X, U, V,True)
    # return np.transpose(V)

    if init != 'random':
        # to use sklearn NMF  on Aug 07, 2024
        from sklearn.decomposition import NMF
        model = NMF(n_components=K, init=init, max_iter=20000) #, random_state=0)
        W = model.fit_transform(X)
        H = model.components_
        return np.transpose(H)

    flag_Repeat = 0
    for repeat in range(1, 1 + nRepeat):
        flag_Repeat = 0
        print(f'\n Starting {repeat}-th repetition\n', file=logFile, flush=True)

        # Initialize U and V
        mean_X = np.sum(X) / (dim_time*dim_space)
        U = (np.random.rand(dim_time, K) + 1) * (np.sqrt(mean_X/K))
        V = (np.random.rand(dim_space, K) + 1) * (np.sqrt(mean_X/K))

        # Normalize data
        U, V = normalize_u_v(U, V, 1, 1, dataPrecision)

        if ard > 0:
            ard = 1
            eta = 0.1
            lambdas = np.sum(U, axis=0) / dim_time
            hyperLam = eta * np.sum(np.power(X, 2)) / (dim_time * dim_space * 2)
        else:
            lambdas = 0
            hyperLam = 0

        oldLogL = np.inf

        # Multiplicative update of U and V
        for i in range(1, 1+maxIter):
            # ===================== update V ========================
            # Eq. 8-11
            XU = X.T @ U
            UU = U.T @ U
            VUU = V @ UU

            tmpl2 = np.power(V, 2)

            if alphaS > 0:
                tmpl21 = np.sqrt(tmpl2)
                tmpl22 = np.tile(np.sqrt(np.sum(tmpl2, axis=0)), (dim_space, 1))
                tmpl21s = np.tile(np.sum(tmpl21, axis=0), (dim_space, 1))
                posTerm = V / np.maximum(tmpl21 * tmpl22, np_eps)
                negTerm = V * tmpl21s / np.maximum(np.power(tmpl22, 3), np_eps)

                VUU = VUU + 0.5 * alphaS * posTerm
                XU = XU + 0.5 * alphaS * negTerm

            if alphaL > 0:
                WV = W @ V.astype(np.float64)
                DV = D @ V.astype(np.float64)

                XU = XU + WV
                VUU = VUU + DV

            V = V * (XU / np.maximum(VUU, np_eps))

            # Prune V if empty components are found in V
            # This is almost impossible to happen without combining FNs
            prunInd = np.sum(V != 0, axis=0) == 1
            if np.any(prunInd):
                V[:, prunInd] = np.zeros((dim_space, np.sum(prunInd)))
                U[:, prunInd] = np.zeros((dim_time, np.sum(prunInd)))

            # normalize U and V
            U, V = normalize_u_v(U, V, 1, 1)

            # ===================== update U =========================
            XV = X @ V
            VV = V.T @ V
            UVV = U @ VV

            if ard > 0:  # ard term for U
                posTerm = 1 / np.maximum(np.tile(lambdas, (dim_time, 1)), np_eps)

                UVV = UVV + posTerm * hyperLam

            U = U * (XV / np.maximum(UVV, np_eps))

            # Prune U if empty components are found in U
            # This is almost impossible to happen without combining FNs
            prunInd = np.sum(U, axis=0) == 0
            if np.any(prunInd):
                V[:, prunInd] = np.zeros((dim_space, np.sum(prunInd)))
                U[:, prunInd] = np.zeros((dim_time, np.sum(prunInd)))

            # update lambda
            if ard > 0:
                lambdas = np.sum(U) / dim_time

            # ==== calculate objective function value ====
            ardU = 0
            tmp2 = 0
            tmpl21 = np.power(V, 2)

            if ard > 0:
                su = np.sum(U, axis=0)
                su[su == 0] = 1
                ardU = np.sum(np.log(su)) * dim_time * hyperLam

            if alphaL > 0:
                tmp2 = V.T @ L * V.T

            L21 = alphaS * np.sum(np.sum(np.sqrt(tmpl21), axis=0) / np.maximum(np.sqrt(np.sum(tmpl21, axis=0)), np_eps))
            # LDf = data_fitting_error(X, U, V, 0, 1)
            LDf = np.sum(np.power(X - U @ V.T, 2))
            LSl = np.sum(tmp2)

            # Objective function
            LogL = L21 + LDf + LSl + ardU
            print(f"    Iter = {i}: LogL: {LogL}, dataFit: {LDf}, spaLap: {LSl}, L21: {L21}, ardU: {ardU}", file=logFile, flush=True)

            # The iteration needs to meet minimum iteration number and small changes of LogL
            if 1 < i < minIter and abs(oldLogL - LogL) / np.maximum(oldLogL, np_eps) < error:
                flag_Repeat = 1
                print('\n Iteration stopped before the minimum iteration number. The results might be poor.\n', file=logFile, flush=True)
                break
            elif i > minIter and abs(oldLogL - LogL) / np.maximum(oldLogL, np_eps) < error:
                break
            oldLogL = LogL.copy()
        if flag_Repeat == 0:
            break

    if flag_Repeat == 1:
        print('\n All repetition stopped before the minimum iteration number. The final results might be poor\n', file=logFile, flush=True)

    gFN = V
    print(f'\nFinished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    return gFN


def gFN_fusion_NCut(gFN_BS, K, NCut_MaxTrial=100, dataPrecision='double', logFile='Log_gFN_fusion_NCut'):
    """
    gFN_fusion_NCut(gFN_BS, K, NCut_MaxTrial=100, dataPrecision='double')
    Fuses FN results to generate representative group-level FNs

    :param gFN_BS: FNs obtained from bootstrapping method, FNs are concatenated along the K dimension
    :param K: Number of FNs, not the total number of FNs obtained from bootstrapping
    :param NCut_MaxTrial: Max number trials for NCut method
    :param dataPrecision: 'double' or 'single'
    :param logFile: str, directory of a txt log file
    :return: gFNs, 2D matrix [dim_space, K]

    Yuncong Ma, 10/2/2023
    """

    # setup log file
    if isinstance(logFile, str):
        logFile = open(logFile, 'a')
    print(f'\nStart NCut for gFN fusion using NumPy at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # Setup data precision and eps
    np_float, np_eps = set_data_precision(dataPrecision)
    if not isinstance(gFN_BS, np.ndarray):
        gFN_BS = np.array(gFN_BS, dype=np_float)
    else:
        gFN_BS = gFN_BS.astype(np_float)

    # clustering by NCut

    # Get similarity between samples
    corrVal = mat_corr(gFN_BS, dataPrecision=dataPrecision)  # similarity between FNs, [K * n_BS, K * n_BS]
    corrVal[np.isnan(corrVal)] = -1
    nDis = 1 - corrVal  # Transform Pearson correlation to non-negative values similar to distance
    triuInd = np.triu(np.ones(nDis.shape), 1)  # Index of upper triangle
    nDisVec = nDis[triuInd == 1]  # Take the values in upper triangle
    # Make all distance non-negative and normalize their distribution
    nW = np.exp(-np.power(nDis, 2) / (np.power(np.median(nDisVec), 2)))  # Transform distance values using exp(-X/std^2) with std as the median value
    nW[np.isnan(nW)] = 0
    sumW = np.sum(nW, axis=1)  # total distance for each FN
    sumW[sumW == 0] = 1  # In case two FNs are the same
    # Construct Laplacian matrix
    D = np.diag(sumW)
    L = np.sqrt(np.linalg.inv(D)) @ nW @ np.linalg.inv(np.sqrt(D))  # A way to normalize nW based on the total distance of each FN
    L = (L + L.T) / 2  # Ensure L is symmetric. Computation error may result in asymmetry

    eVal, Ev = scipy.sparse.linalg.eigs(L.astype(np.float64), K, which='LR')  # Get first K eigenvectors, sign of vectors may be different to MATLAB results
    Ev = np.real(Ev)
    # Correct the sign of eigenvectors to make them same as derived from MATLAB
    temp = np.sign(np.sum(Ev, axis=0))  # Use the total value of each eigenvector to reset its sign
    temp[temp == 0.0] = 1.0
    Ev = Ev * np.tile(temp, (Ev.shape[0], 1))  # Reset the sign of each eigenvector
    normvect = np.sqrt(np.diag(Ev @ Ev.T))  # Get the norm of each eigenvector
    normvect[normvect == 0.0] = 1  # Incase all 0 eigenvector
    Ev = np.linalg.solve(np.diag(normvect), Ev)  # Use linear solution to normalize Ev satisfying normvect * Ev_new = Ev_old

    # Multiple trials to get reproducible results
    Best_C = []
    Best_NCutValue = np.inf
    for i in range(NCut_MaxTrial):
        print(f'    Iter = ' + str(i), file=logFile)
        EigenVectors = Ev
        n, k = EigenVectors.shape  # n is K * n_BS, k is K

        vm = np.sqrt(np.sum(EigenVectors**2, axis=1, keepdims=True))  # norm of each row
        EigenVectors = EigenVectors / np.tile(vm, (1, k))  # normalize eigenvectors to ensure each FN vector's norm = 1

        R = np.zeros((k, k))
        ps = np.random.randint(0, n, 1)  # Choose a random row in eigenvectors
        R[:, 0] = EigenVectors[ps, :]  # This randomly selected row in eigenvectors is used as an initial center

        c_index = np.zeros(n)
        c = np.zeros(n)  # Total distance to different rows in R [K * n_BS, 1]
        c_index[0] = ps  # Store the index of selected samples
        c[ps] = np.inf

        for j in range(2, k+1):  # Find another K-1 rows in eigenvectors which have the minimum similarity to previous selected rows, similar to initialization in k++
            c += np.abs(EigenVectors @ R[:, j-2])
            ps = np.argmin(c)
            c_index[j-1] = ps
            c[ps] = np.inf
            R[:, j-1] = EigenVectors[ps, :].T

        lastObjectiveValue = 0
        exitLoop = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        while exitLoop == 0:
            nbIterationsDiscretisation += 1

            EigenVectorsR = EigenVectors @ R
            n, k = EigenVectorsR.shape
            J = np.argmax(EigenVectorsR, axis=1)  # Assign each sample to K centers of R based on highest similarity
            EigenvectorsDiscrete = scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), J)), shape=(n, k))  # Generate a 0-1 matrix with each row containing only one 1

            U, S, Vh = scipy.linalg.svd(EigenvectorsDiscrete.T @ EigenVectors, full_matrices=False)  # Economy-size decomposition

            S = np.diag(S)  # To match MATLAB svd results
            V = Vh.T  # To match MATLAB svd results
            NcutValue = 2 * (n - np.trace(S))

            # escape the loop when converged or meet max iteration
            if abs(NcutValue - lastObjectiveValue) < np_eps or nbIterationsDiscretisation > nbIterationsDiscretisationMax:
                exitLoop = 1
                print(f'    Reach stop criterion of NCut, NcutValue = '+str(NcutValue)+'\n', file=logFile, flush=True)
            else:
                print(f'    NcutValue = '+str(NcutValue), file=logFile)
                lastObjectiveValue = NcutValue
                R = V @ U.T  # Update R which stores the new centers

        C = np.argmax(EigenvectorsDiscrete.toarray(), axis=1)  # Assign each sample to K centers in R

        if len(np.unique(C)) < K:  # Check whether there are empty results
            print(f'    Found empty results in iteration '+str(i+1)+'\n', file=logFile, flush=True)
        else:  # Update the best result
            if NcutValue < Best_NCutValue:
                Best_NCutValue = NcutValue
                Best_C = C

    if len(set(Best_C)) < K:  # In case even the last trial has empty results
        raise ValueError('  Cannot generate non-empty gFNs\n')
        Flag = 1
        Message = "Cannot generate non-empty FN"

    print(f'Best NCut value = '+str(Best_NCutValue)+'\n', file=logFile, flush=True)

    # Get centroid
    C = Best_C
    gFN = np.zeros((gFN_BS.shape[0], K))
    for ki in range(K):
        if np.sum(C == ki) > 1:
            candSet = gFN_BS[:, C == ki]  # Get the candidate set of FNs assigned to cluster ki
            corrW = np.abs(mat_corr(candSet, dataPrecision=dataPrecision))  # Get the similarity between candidate FNs
            corrW[np.isnan(corrW)] = 0
            mInd = np.argmax(np.sum(corrW, axis=0), axis=0)  # Find the FN with the highest total similarity to all other FNs
            gFN[:, ki] = candSet[:, mInd]
        elif np.sum(C == ki) == 1:
            mInd = C.tolist().index(ki)
            gFN[:, ki] = gFN_BS[:, mInd]

    gFN = gFN / np.maximum(np.tile(np.max(gFN, axis=0), (gFN.shape[0], 1)), np_eps)  # Normalize each FN by its max value
    print(f'\nFinished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    return gFN


def compute_gNb(Brain_Template, logFile=None):
    """
    compute_gNb(Brain_Template, logFile=None)
    Prepare a graph neighborhood variable, using indices as its sparse representation

    :param Brain_Template: a structure variable with keys 'Data_Type', 'Template_Format', 'Shape', 'Brain_Mask'.
        If Brain_Template.Data_Type is 'Surface', Shape contains L and R, with vertices and faces as sub keys. Brain_Mask contains L and R.
        If Brain_Template.Data_Type is 'Volume', Shape is None, Brain_Mask is a 3D 0-1 matrix, Overlay_Image is a 3D matrix
        If Brain_Template.Data_Type is 'Surface-Volume', It includes fields from both 'Surface' and 'Volume', 'Brain_Mask' is renamed to be 'Surface_Mask' and 'Volume_Mask'
    :param logFile:
    :return: gNb: a 2D matrix [N, 2], which labels the non-zero elements in a graph. Index starts from 1

    Yuncong Ma, 11/13/2023
    """

    # Check Brain_Template
    if 'Data_Type' not in Brain_Template.keys():
        raise ValueError('Cannot find Data_Type in the Brain_Template')
    if 'Template_Format' not in Brain_Template.keys():
        raise ValueError('Cannot find Data_Format in the Brain_Template')

    # Construct gNb
    if Brain_Template['Data_Type'] == 'Surface':
        Brain_Surface = Brain_Template
        # Number of vertices
        Nv_L = Brain_Surface['Shape']['L']['vertices'].shape[0]  # left hemisphere
        Nv_R = Brain_Surface['Shape']['R']['vertices'].shape[0]
        # Number of faces
        Nf_L = Brain_Surface['Shape']['L']['faces'].shape[0]  # left hemisphere
        Nf_R = Brain_Surface['Shape']['R']['faces'].shape[0]
        # Index of useful vertices, starting from 1
        vL = np.sort(np.where(Brain_Surface['Brain_Mask']['L'] == 1)[0]) + int(1)  # left hemisphere
        vR = np.sort(np.where(Brain_Surface['Brain_Mask']['R'] == 1)[0]) + int(1)
        # Create gNb using matrix format
        # Exclude the medial wall or other vertices outside the mask
        # Set the maximum size to avoid unnecessary memory allocation
        gNb_L = np.zeros((3 * Nf_L, 2), dtype=np.int64)
        gNb_R = np.zeros((3 * Nf_R, 2), dtype=np.int64)
        Count_L = 0
        Count_R = 0

        # Get gNb of all useful vertices in the left hemisphere
        for i in range(0, Nf_L):
            temp = Brain_Surface['Shape']['L']['faces'][i]
            temp = np.intersect1d(temp, vL)

            if len(temp) == 2:  # only one line
                gNb_L[Count_L, :] = temp
                Count_L += 1
            elif len(temp) == 3:  # three lines
                temp = np.tile(temp, (2, 1)).T
                temp[:, 1] = temp[(1, 2, 0), 1]
                gNb_L[Count_L:Count_L + 3, :] = temp
                Count_L += 3
            else:
                continue
        gNb_L = gNb_L[0:Count_L, :]  # Remove unused part
        # Right hemisphere
        for i in range(0, Nf_R):
            temp = Brain_Surface['Shape']['R']['faces'][i]
            temp = np.intersect1d(temp, vR)

            if len(temp) == 2:  # only one line
                gNb_R[Count_R, :] = temp
                Count_R += 1
            elif len(temp) == 3:  # three lines
                temp = np.tile(temp, (2, 1)).T
                temp[:, 1] = temp[(1, 2, 0), 1]
                gNb_R[Count_R:Count_R + 3, :] = temp
                Count_R += 3
            else:
                continue
        gNb_R = gNb_R[0:Count_R, :]  # Remove unused part

        # Map the index from all vertices to useful ones
        mapL = np.zeros(Nv_L, dtype=np.int64)
        mapL[vL - 1] = range(1, 1+len(vL))  # Python index starts from 0, gNb index starts from 1
        gNb_L = mapL[(gNb_L.flatten() - 1).astype(int)]
        gNb_L = np.reshape(gNb_L, (int(np.round(len(gNb_L)/2)), 2))
        gNb_L = np.append(gNb_L, gNb_L[:, (-1, 0)], axis=0)
        # right hemisphere
        mapR = np.zeros(Nv_R, dtype=np.int64)
        mapR[vR - 1] = range(1, 1+len(vR))
        gNb_R = mapR[(gNb_R.flatten() - 1).astype(int)]
        gNb_R = np.reshape(gNb_R, (int(np.round(len(gNb_R)/2)), 2))
        gNb_R = np.append(gNb_R, gNb_R[:, (-1, 0)], axis=0)

        # Combine two hemispheres
        gNb = np.concatenate((gNb_L, gNb_R + len(vL)), axis=0)  # Shift index in right hemisphere by the number of useful vertices in left hemisphere
        gNb = np.unique(gNb, axis=0)  # Remove duplicated

    elif Brain_Template['Data_Type'] == 'Volume':
        Brain_Mask = Brain_Template['Brain_Mask'] > 0
        if not (len(Brain_Mask.shape) == 3 or (len(Brain_Mask.shape) == 4 and Brain_Mask.shape[3] == 1)):
            raise ValueError('Mask in Brain_Template needs to be a 3D matrix when the data type is volume')
        if len(Brain_Mask.shape) == 4:
            Brain_Mask = np.squeeze(Brain_Mask, axis=3)

        # Index starts from 1
        sx = Brain_Mask.shape[0]
        sy = Brain_Mask.shape[1]
        sz = Brain_Mask.shape[2]
        # Label non-zero elements in Brain_Mask
        Nm = np.sum(Brain_Mask > 0)
        maskLabel = Brain_Mask.flatten('F')
        maskLabel = maskLabel.astype(np.int64)  # Brain_Mask might be a 0-1 logic matrix
        if 'Volume_Order' in Brain_Template.keys():  # customized index order
            maskLabel[maskLabel > 0] = Brain_Template['Volume_Order']
        else:  # default index order
            maskLabel[maskLabel > 0] = range(1, 1 + Nm)
        maskLabel = np.reshape(maskLabel, Brain_Mask.shape, order='F')  # consistent to MATLAB matrix index order
        # Enumerate each voxel in Brain_Mask
        # The following code is optimized for NumPy array
        # Set the maximum size to avoid unnecessary memory allocation
        gNb = np.zeros((Nm * 26, 2), dtype=np.int64)
        Count = 0
        for xi in range(sx):
            for yi in range(sy):
                for zi in range(sz):
                    if Brain_Mask[xi, yi, zi] > 0:
                        Brain_Mask[xi, yi, zi] = 0  # Exclude the self point
                        #  Create a 3x3x3 box, +2 is for Python range
                        patchBox = (np.maximum((xi - 1, yi - 1, zi - 1), (0, 0, 0)), np.minimum((xi + 2, yi + 2, zi + 2), (sx, sy, sz)))
                        for xni in range(patchBox[0][0], patchBox[1][0]):
                            for yni in range(patchBox[0][1], patchBox[1][1]):
                                for zni in range(patchBox[0][2], patchBox[1][2]):
                                    if Brain_Mask[xni, yni, zni] > 0:
                                        gNb[Count, :] = (maskLabel[xi, yi, zi], maskLabel[xni, yni, zni])
                                        Count += 1
                        Brain_Mask[xi, yi, zi] = 1  # reset the self value

        gNb = gNb[0:Count, :]  # Remove unused space
        gNb = np.unique(gNb, axis=0)  # Remove duplicated, and sort the results

    elif Brain_Template['Data_Type'] == 'Surface-Volume':
        Brain_Template_surf = Brain_Template.copy()
        Brain_Template_surf['Data_Type'] = 'Surface'
        Brain_Template_surf['Brain_Mask'] = Brain_Template_surf['Surface_Mask']
        gNb_surf = compute_gNb(Brain_Template_surf, logFile=logFile)
        Brain_Template_vol = Brain_Template.copy()
        Brain_Template_vol['Data_Type'] = 'Volume'
        Brain_Template_vol['Brain_Mask'] = Brain_Template_surf['Volume_Mask']
        gNb_vol = compute_gNb(Brain_Template_vol, logFile=logFile)
        # Concatenate the two gNbs with index adjustment
        gNb = np.concatenate((gNb_surf, gNb_vol + np.max(gNb_surf)), axis=0)

    else:
        raise ValueError('Unknown combination of Data_Type and Data_Surface: ' + Brain_Template['Data_Type'] + ' : ' + Brain_Template['Data_Format'])

    if len(np.unique(gNb[:, 0])) != max(np.unique(gNb[:, 0])):
        if logFile is not None:
            if isinstance(logFile, str):
                logFile = open(logFile, 'a')
            print('\ngNb contains isolated voxel or vertex which will affect the subsequent analysis', file=logFile, flush=True)

    if logFile is not None:
        print('\ngNb is generated successfully', file=logFile, flush=True)

    return gNb


def bootstrap_scan(dir_output: str, file_scan: str, file_subject_ID: str, file_subject_folder: str, file_group_ID=None, combineScan=0,
                   samplingMethod='Subject', sampleSize=10, nBS=50, logFile=None):
    """
    bootstrap_scan(dir_output: str, file_scan: str, file_subject_ID: str, file_subject_folder: str, file_group=None, combineScan=0, samplingMethod='Subject', sampleSize=10, nBS=50, logFile=None)
    prepare bootstrapped scan file lists

    :param dir_output: directory of a folder to store bootstrapped files
    :param file_scan: a txt file that stores directories of all fMRI scans
    :param file_subject_ID: a txt file that store subject ID information corresponding to fMRI scan in file_scan
    :param file_subject_folder: a txt file that store subject folder names corresponding to fMRI scans in file_scan
    :param file_group_ID: a txt file that store group information corresponding to fMRI scan in file_scan
    :param combineScan: 0 or 1, whether to combine multiple fMRI scans for each subject
    :param samplingMethod: 'Subject' or 'Group_Subject'. Uniform sampling based subject ID, or group and then subject ID
    :param sampleSize: number of subjects selected for each bootstrapping run
    :param nBS: number of runs for bootstrap
    :param logFile: directory of a txt file
    :return: None

    Yuncong Ma, 10/2/2023
    """

    if logFile is not None:
        if isinstance(logFile, str):
            logFile = open(logFile, 'a')
        print(f'\nStart preparing bootstrapped scan list files '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # Lists for input
    list_scan = np.array([line.replace('\n', '') for line in open(file_scan, 'r')])
    list_subject_ID = np.array([line.replace('\n', '') for line in open(file_subject_ID, 'r')])
    subject_ID_unique = np.unique(list_subject_ID)
    N_Subject = subject_ID_unique.shape[0]
    list_subject_folder = np.array([line.replace('\n', '') for line in open(file_subject_folder, 'r')])
    if file_group_ID is not None:
        list_group_D = ''
    else:
        list_group_D = None
    if list_group_D is not None:
        group_unique = np.unique(list_subject_ID)

    # check parameter
    if sampleSize > N_Subject: # changed on 08/01/2024
        sampleSize = N_Subject
        #raise ValueError('The number of randomly selected subjects should be no more than the total number of subjects')
    if samplingMethod == 'Group_Subject' and (list_group_D is None or len(list_group_D) == 0):
        raise ValueError('Group information is absent')
    if samplingMethod != 'Subject' and samplingMethod != 'Group_Subject':
        raise ValueError('Unknown sampling method for bootstrapping: ' + samplingMethod)
    for i in range(1, nBS+1):
        if not os.path.exists(os.path.join(dir_output, str(i))):
            os.mkdir(os.path.join(dir_output, str(i)))
        List_BS = np.empty(sampleSize, dtype=list)

        # Randomly select subjects
        if samplingMethod == 'Subject':
            ps = np.sort(np.random.choice(N_Subject, sampleSize, replace=False))
            for j in range(sampleSize):
                if combineScan == 1:
                    # Get all scans from the selected subject
                    temp = list_scan[np.where(np.compare_chararrays(list_subject_ID, subject_ID_unique[ps[j]], '==', False))[0]]
                    List_BS[j] = str.join('\n', temp)
                else:
                    # Choose one scan from the selected subject
                    temp = list_scan[np.where(np.compare_chararrays(list_subject_ID, subject_ID_unique[ps[j]], '==', False))[0]]
                    ps2 = np.random.choice(temp.shape[0], 1)  # list
                    List_BS[j] = temp[ps2[0]]  # transform to string list

        if samplingMethod == 'Group_Subject':
            break

        # Write the Scan_List.txt file
        if logFile is not None:
            print('\nWrite bootstrapped scan list in file ' + os.path.join(dir_output, str(i), 'Scan_List.txt'), file=logFile)
        FID = open(os.path.join(dir_output, str(i), 'Scan_List.txt'), 'w')
        for j in range(sampleSize):
            print(List_BS[j], file=FID)
        FID.close()


def setup_SR_NMF(dir_pnet_result: str, K=17, Combine_Scan=False, file_gFN=None, init='random', samplingMethod='Subject', sampleSize='Automatic', nBS=50, nTPoints=99999, maxIter=(2000, 500), minIter=200, meanFitRatio=0.1, error=1e-8,
                 normW=1, Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, ard=0, eta=0, nRepeat=5, Parallel=False, Computation_Mode='CPU', N_Thread=1, dataPrecision='double', outputFormat='Both'):
    """
    Setup SR-NMF parameters to compute gFNs and pFNs

    :param dir_pnet_result: directory of the pNet result folder
    :param K: number of FNs
    :param Combine_Scan: False or True, whether to combine multiple scans for the same subject
    :param file_gFN: directory of a precomputed gFN in .mat format
    :param init: 'nndsvda': NNDSVD with zeros filled with the average of X (better when sparsity is not desired)  #updated on 08/03/2024
                 'random': non-negative random matrices, scaled with: sqrt(X.mean() / n_components)
                 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness)
                 'nndsvdar' NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to NNDSVDa for when spars
    :param samplingMethod: 'Subject' or 'Group_Subject'. Uniform sampling based subject ID, or group and then subject ID
    :param sampleSize: 'Automatic' or integer number, number of subjects selected for each bootstrapping run
    :param nBS: 'Automatic' or integer number, number of runs for bootstrap
    :param maxIter: maximum iteration number for multiplicative update, which can be one number or two numbers for gFN and pFN separately
    :param minIter: minimum iteration in case fast convergence, which can be one number or two numbers for gFN and pFN separately
    :param meanFitRatio: a 0-1 scaler, exponential moving average coefficient, used for the initialization of U when using group initialized V
    :param error: difference of cost function for convergence
    :param normW: 1 or 2, normalization method for W used in Laplacian regularization
    :param Alpha: hyper parameter for spatial sparsity
    :param Beta: hyper parameter for Laplacian sparsity
    :param alphaS: internally determined, the coefficient for spatial sparsity based Alpha, data size, K, and gNb
    :param alphaL: internally determined, the coefficient for Laplacian sparsity based Beta, data size, K, and gNb
    :param vxI: flag for using the temporal correlation between nodes (vertex, voxel)
    :param ard: 0 or 1, flat for combining similar clusters
    :param eta: a hyper parameter for the ard regularization term
    :param nRepeat: Any positive integer, the number of repetition to avoid poor initialization
    :param Parallel: False or True, whether to enable parallel computation
    :param Computation_Mode: 'CPU'
    :param N_Thread: positive integers, used for parallel computation
    :param dataPrecision: 'double' or 'single'
    :param outputFormat: 'MAT', 'Both', 'MAT' is to save results in FN.mat and TC.mat for functional networks and time courses respectively. 'Both' is for both matlab format and fMRI input file format

    :return: setting: a structure

    Yuncong Ma, 2/2/2024
    """

    dir_pnet_dataInput, dir_pnet_FNC, _, _, _, _ = setup_result_folder(dir_pnet_result)

    # Set sampleSize if it is 'Automatic'
    if sampleSize == 'Automatic':
        file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
        list_subject_ID = np.array([line.replace('\n', '') for line in open(file_subject_ID, 'r')])
        subject_ID_unique = np.unique(list_subject_ID)
        N_Subject = subject_ID_unique.shape[0]
        if sampleSize == 'Automatic':
            sampleSize = np.maximum(100, np.round(N_Subject / 10))
            if N_Subject < sampleSize:  # added by hm
                sampleSize = N_Subject #- 1  #changed by Yong Fan: for sample datasets, all subjects/scans are used
                #nBS = 10   # was 5, changed by Yong Fan

    # add nTPoints on 08/01/2024
    # add init on 08/03/2024
    BootStrap = {'samplingMethod': samplingMethod, 'sampleSize': sampleSize, 'nBS': nBS, 'nTPoints': nTPoints, 'init': init}
    Group_FN = {'file_gFN': file_gFN,
                'BootStrap': BootStrap,
                'maxIter': maxIter, 'minIter': minIter, 'error': error,
                'normW': normW, 'Alpha': Alpha, 'Beta': Beta, 'alphaS': alphaS, 'alphaL': alphaL, 'vxI': vxI,
                'ard': ard, 'eta': eta, 'nRepeat': nRepeat}
    Personalized_FN = {'maxIter': maxIter, 'minIter': minIter, 'meanFitRatio': meanFitRatio, 'error': error,
                       'normW': normW, 'Alpha': Alpha, 'Beta': Beta, 'alphaS': alphaS, 'alphaL': alphaL,
                       'vxI': vxI, 'ard': ard, 'eta': eta}
    Computation = {'Parallel': Parallel,
                   'Model': Computation_Mode,
                   'N_Thread': N_Thread,
                   'dataPrecision': dataPrecision}

    setting = {'Method': 'SR-NMF',
               'K': K,
               'Combine_Scan': Combine_Scan,
               'Group_FN': Group_FN,
               'Personalized_FN': Personalized_FN,
               'Computation': Computation,
               'Output_Format': outputFormat}

    write_json_setting(setting, os.path.join(dir_pnet_FNC, 'Setting.json'))
    return setting


def setup_pFN_folder(dir_pnet_result: str):
    """
    setup_pFN_folder(dir_pnet_result: str)
    Setup sub-folders in Personalized_FN to

    :param dir_pnet_result: directory of the pNet result folder
    :return: list_subject_folder_unique: unique subject folder array for getting sub-folders in Personalized_FN

    Yuncong Ma, 9/25/2023
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, _, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # load settings for data input and FN computation
    if not os.path.isfile(os.path.join(dir_pnet_FNC, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder FN_Computation')
    setting = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))

    combineScan = setting['Combine_Scan']

    file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')

    list_scan = [line.replace('\n', '') for line in open(file_scan, 'r')]
    list_subject_folder = [line.replace('\n', '') for line in open(file_subject_folder, 'r')]
    list_subject_folder = np.array(list_subject_folder)
    list_subject_folder_unique = np.unique(list_subject_folder)

    # Check consistency of setting and files
    if len(list_subject_folder) != len(list_scan):
        raise ValueError('The length of contents in Scan_List.txt and Subject_Folder.txt does NOT match')
    if combineScan and len(list_subject_folder_unique) == len(list_subject_folder):
        raise ValueError('When combineScan is enabled, the txt file Subject_Folder.txt is supposed to show repeated sub-folder names')

    N_Scan = list_subject_folder_unique.shape[0]
    for i in range(N_Scan):
        template = list_subject_folder_unique[i]
        # find scan indexes that match to the subject folder
        scan_index = [i for i, x in enumerate(list_subject_folder) if x == template]
        dir_pnet_pFN_indv = os.path.join(dir_pnet_pFN, template)
        if not os.path.exists(dir_pnet_pFN_indv):
            os.makedirs(dir_pnet_pFN_indv)
        file_scan_ind = os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt')
        file_scan_ind = open(file_scan_ind, 'w')
        for j in range(len(scan_index)):
            print(list_scan[scan_index[j]] + '\n', file=file_scan_ind)
        file_scan_ind.close()

    return list_subject_folder_unique


def run_FN_Computation(dir_pnet_result: str):
    """
    run_FN_Computation(dir_pnet_result: str)
    run the FN Computation module with settings ready in Data_Input and FN_Computation

    :param dir_pnet_result: directory of pNet result folder

    Yuncong Ma, 1/11/2024
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # log file
    logFile_FNC = os.path.join(dir_pnet_FNC, 'log.log')
    logFile_FNC = open(logFile_FNC, 'w')
    print('\nStart FN computation using Numpy at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=logFile_FNC, flush=True)

    # load settings for data input and FN computation
    if not os.path.isfile(os.path.join(dir_pnet_dataInput, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder Data_Input')
    if not os.path.isfile(os.path.join(dir_pnet_FNC, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder FN_Computation')
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC}
    print('Settings are loaded from folder Data_Input and FN_Computation', file=logFile_FNC, flush=True)

    # load basic settings
    dataType = setting['Data_Input']['Data_Type']
    dataFormat = setting['Data_Input']['Data_Format']

    # load Brain Template
    Brain_Template = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))

    if dataType == 'Volume':
        Brain_Mask = Brain_Template['Brain_Mask']
    else:
        Brain_Mask = None
    print('Brain template is loaded from folder Data_Input', file=logFile_FNC, flush=True)

    # ============== gFN Computation ============== #
    # Start computation using SP-NMF
    if setting['FN_Computation']['Method'] == 'SR-NMF':
        print('FN computation uses sparsity-regularized non-negative matrix factorization method', file=logFile_FNC, flush=True)

        # Generate additional parameters
        gNb = compute_gNb(Brain_Template)
        scipy.io.savemat(os.path.join(dir_pnet_FNC, 'gNb.mat'), {'gNb': gNb}, do_compression=True)

        if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
            # 2 steps
            # step 1 ============== bootstrap
            # sub-folder in FNC for storing bootstrapped results
            print('Start to prepare bootstrap files', file=logFile_FNC, flush=True)
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            if not os.path.exists(dir_pnet_BS):
                os.makedirs(dir_pnet_BS)
            # Log
            logFile = os.path.join(dir_pnet_BS, 'Log.log')

            # Input files
            file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
            file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
            file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
            file_group_ID = os.path.join(dir_pnet_dataInput, 'Group_ID.txt')
            if not os.path.exists(file_group_ID):
                file_group = None
            # Parameters
            combineScan = setting['FN_Computation']['Combine_Scan']
            init = setting['FN_Computation']['Group_FN']['BootStrap']['init']  # added on 08/03/2024
            samplingMethod = setting['FN_Computation']['Group_FN']['BootStrap']['samplingMethod']
            sampleSize = setting['FN_Computation']['Group_FN']['BootStrap']['sampleSize']
            nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']
            nTPoints = setting['FN_Computation']['Group_FN']['BootStrap']['nTPoints']  #added on 08/01/2024

            # create scan lists for bootstrap
            bootstrap_scan(dir_pnet_BS, file_scan, file_subject_ID, file_subject_folder,
                           file_group_ID=file_group_ID, combineScan=combineScan,
                           samplingMethod=samplingMethod, sampleSize=sampleSize, nBS=nBS, logFile=logFile)

            # Parameters
            K = setting['FN_Computation']['K']
            maxIter = setting['FN_Computation']['Group_FN']['maxIter']
            minIter = setting['FN_Computation']['Group_FN']['minIter']
            error = setting['FN_Computation']['Group_FN']['error']
            normW = setting['FN_Computation']['Group_FN']['normW']
            Alpha = setting['FN_Computation']['Group_FN']['Alpha']
            Beta = setting['FN_Computation']['Group_FN']['Beta']
            alphaS = setting['FN_Computation']['Group_FN']['alphaS']
            alphaL = setting['FN_Computation']['Group_FN']['alphaL']
            vxI = setting['FN_Computation']['Group_FN']['vxI']
            ard = setting['FN_Computation']['Group_FN']['ard']
            eta = setting['FN_Computation']['Group_FN']['eta']
            nRepeat = setting['FN_Computation']['Group_FN']['nRepeat']
            dataPrecision = setting['FN_Computation']['Computation']['dataPrecision']

            # separate maxIter and minIter for gFN and pFN
            if isinstance(maxIter, int) or (isinstance(maxIter, numpy.ndarray) and maxIter.shape == 1):
                maxIter_gFN = maxIter
                maxIter_pFN = maxIter
            else:
                maxIter_gFN = maxIter[0]
                maxIter_pFN = maxIter[1]
            if isinstance(minIter, int) or (isinstance(minIter, numpy.ndarray) and minIter.shape == 1):
                minIter_gFN = minIter
                minIter_pFN = minIter
            else:
                minIter_gFN = minIter[0]
                minIter_pFN = minIter[1]

            # NMF on bootstrapped subsets
            print('Start to NMF for each bootstrap at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)
            for rep in range(1, 1+nBS):
                # log file
                logFile = os.path.join(dir_pnet_BS, str(rep), 'Log.log')
                # load data
                file_scan_list = os.path.join(dir_pnet_BS, str(rep), 'Scan_List.txt')
                Data,CHeader,NHeader = load_fmri_scan(file_scan_list, dataType=dataType, dataFormat=dataFormat, nTPoints=nTPoints, Reshape=True, Brain_Mask=Brain_Mask,
                                      Normalization='vp-vmax', logFile=logFile)
                # perform NMF
                FN_BS = gFN_NMF(Data, K, gNb, init=init, maxIter=maxIter_gFN, minIter=minIter_gFN, error=error, normW=normW,
                                Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL, vxI=vxI, ard=ard, eta=eta,
                                nRepeat=nRepeat, dataPrecision=dataPrecision, logFile=logFile)
                # save results
                FN_BS = reshape_FN(FN_BS, dataType=dataType, Brain_Mask=Brain_Mask)
                sio.savemat(os.path.join(dir_pnet_BS, str(rep), 'FN.mat'), {"FN": FN_BS}, do_compression=True)
                # save FNs in nii.gz and TC as txt file  FY 07/26/2024
                output_FN(FN=FN_BS, 
                          file_output=os.path.join(dir_pnet_BS, str(rep), 'FN.mat'), 
                          file_brain_template = Brain_Template, 
                          dataFormat=dataFormat,
                          Cheader = CHeader,
                          Nheader = NHeader)

            # step 2 ============== fuse results
            # Generate gFNs
            print('Start to fuse bootstrapped results using NCut at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)
            FN_BS = np.empty(nBS, dtype=np.ndarray)
            # load bootstrapped results
            for rep in range(1, nBS+1):
                FN_BS[rep-1] = np.array(reshape_FN(load_matlab_single_array(os.path.join(dir_pnet_BS, str(rep), 'FN.mat')), dataType=dataType, Brain_Mask=Brain_Mask))
            gFN_BS = np.concatenate(FN_BS, axis=1)
            # log
            logFile = os.path.join(dir_pnet_gFN, 'Log.log')
            # Fuse bootstrapped results
            gFN = gFN_fusion_NCut(gFN_BS, K, logFile=logFile)
            # output
            gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
            # save FNs in nii.gz and TC as txt file  FY 07/26/2024
            output_FN(FN=gFN,
                      file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
                      file_brain_template = Brain_Template,
                      dataFormat=dataFormat,
                      Cheader = CHeader,
                      Nheader = NHeader)

        else:  # use precomputed gFNs
            file_gFN = setting['FN_Computation']['Group_FN']['file_gFN']
            gFN = load_matlab_single_array(file_gFN)
            if dataType == 'Volume':
                Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
                gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            check_gFN(gFN, method=setting['FN_Computation']['Method'])
            if dataType == 'Volume':
                gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
            # save FNs in nii.gz and TC as txt file  FY 07/26/2024
            #output_FN(FN=gFN,
            #          file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
            #          file_brain_template = Brain_Template,
            #          dataFormat=dataFormat, Cheader = CHeader, Nheader = NHeader)

            print('load precomputed gFNs', file=logFile_FNC, flush=True)
        # ============================================= #

        # ============== pFN Computation ============== #
        print('Start to compute pFNs at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)
        # load precomputed gFNs
        gFN = load_matlab_single_array(os.path.join(dir_pnet_gFN, 'FN.mat'))
        # additional parameter
        gNb = load_matlab_single_array(os.path.join(dir_pnet_FNC, 'gNb.mat'))
        # reshape to 2D if required
        gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
        # setup folders in Personalized_FN
        list_subject_folder = setup_pFN_folder(dir_pnet_result)
        N_Scan = len(list_subject_folder)
        for i in range(1, N_Scan+1):
            print(f'Start to compute pFNs for {i}-th folder: {list_subject_folder[i-1]}', file=logFile_FNC, flush=True)
            dir_pnet_pFN_indv = os.path.join(dir_pnet_pFN, list_subject_folder[i-1])
            # parameter
            maxIter = setting['FN_Computation']['Personalized_FN']['maxIter']
            minIter = setting['FN_Computation']['Personalized_FN']['minIter']
            meanFitRatio = setting['FN_Computation']['Personalized_FN']['meanFitRatio']
            error = setting['FN_Computation']['Personalized_FN']['error']
            normW = setting['FN_Computation']['Personalized_FN']['normW']
            Alpha = setting['FN_Computation']['Personalized_FN']['Alpha']
            Beta = setting['FN_Computation']['Personalized_FN']['Beta']
            alphaS = setting['FN_Computation']['Personalized_FN']['alphaS']
            alphaL = setting['FN_Computation']['Personalized_FN']['alphaL']
            vxI = setting['FN_Computation']['Personalized_FN']['vxI']
            ard = setting['FN_Computation']['Personalized_FN']['ard']
            eta = setting['FN_Computation']['Personalized_FN']['eta']
            dataPrecision = setting['FN_Computation']['Computation']['dataPrecision']

            # separate maxIter and minIter for gFN and pFN
            if isinstance(maxIter, int) or (isinstance(maxIter, numpy.ndarray) and maxIter.shape == 1):
                maxIter_gFN = maxIter
                maxIter_pFN = maxIter
            else:
                maxIter_gFN = maxIter[0]
                maxIter_pFN = maxIter[1]
            if isinstance(minIter, int) or (isinstance(minIter, numpy.ndarray) and minIter.shape == 1):
                minIter_gFN = minIter
                minIter_pFN = minIter
            else:
                minIter_gFN = minIter[0]
                minIter_pFN = minIter[1]

            # log file
            logFile = os.path.join(dir_pnet_pFN_indv, 'Log.log')
            # load data
            Data, CHeader, NHeader = load_fmri_scan(os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt'),
                                  dataType=dataType, dataFormat=dataFormat,
                                  Reshape=True, Brain_Mask=Brain_Mask, logFile=logFile)
            # perform NMF
            TC, pFN = pFN_NMF(Data, gFN, gNb, maxIter=maxIter_pFN, minIter=minIter_pFN, meanFitRatio=meanFitRatio, error=error, normW=normW,
                              Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL, vxI=vxI, ard=ard, eta=eta,
                              dataPrecision=dataPrecision, logFile=logFile)
            # output
            # pFN = reshape_FN(pFN.numpy(), dataType=dataType, Brain_Mask=Brain_Mask)  updated on 08/02/2024 removed .numpy()
            pFN = reshape_FN(pFN, dataType=dataType, Brain_Mask=Brain_Mask)
            sio.savemat(os.path.join(dir_pnet_pFN_indv, 'FN.mat'), {"FN": pFN}, do_compression=True)
            sio.savemat(os.path.join(dir_pnet_pFN_indv, 'TC.mat'), {"TC": TC}, do_compression=True)
            # save FNs in nii.gz and TC as txt file  FY 07/26/2024
            output_FN(FN=pFN,
                      file_output=os.path.join(dir_pnet_pFN_indv, 'FN.mat'),
                      file_brain_template = Brain_Template,
                      dataFormat=dataFormat, 
                      Cheader = CHeader,
                      Nheader = NHeader)
            np.savetxt(os.path.join(dir_pnet_pFN_indv, 'TC.txt'), TC)

        # ============================================= #

    print('Finished FN computation at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)


def check_gFN(gFN: np.ndarray, method='SR-NMF', logFile=None):
    """
    Check the values in gFNs to ensure compatibility to the desired FN model

    :param gFN: group level FNs, 2D matrix for surface type [V K], 4D matrix for volume type [X Y Z K], where K is the number of FNs
    :param method: 'SR-NMF'
    :param logFile: directory of the log file

    Yuncong Ma, 9/27/2023
    """

    if method == 'SR-NMF':
        if np.sum(gFN < 0) > 0:
            raise ValueError('When using method SR-NMF, all values in gFNs should be non-negative')
            if logFile is not None:
                logFile = open(logFile, 'a')
                print('When using method SR-NMF, all values in gFNs should be non-negative', file=logFile, flush=True)
