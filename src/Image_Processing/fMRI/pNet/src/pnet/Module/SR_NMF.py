# Yuncong Ma, 2/12/2024
# SR-NMF method in pNet
# Pytorch version
# To avoid same function naming, use import SR_NMF

#########################################
# Packages


# other functions of pNet
from Module.Data_Input import *
from Basic.Matrix_Computation import *


def setup_SR_NMF(dir_pnet_result: str or None, K=17, Combine_Scan=False,
                 file_gFN=None, init='random', samplingMethod='Subject', sampleSize='Automatic', nBS=50, nTPoints=99999,
                 maxIter=(2000, 500), minIter=200, meanFitRatio=0.1, error=1e-8,
                 normW=1, Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, ard=0, eta=0, nRepeat=5,
                 Parallel=False, Computation_Mode='CPU', N_Thread=1, dataPrecision='double', outputFormat='Both'):
    """
    Setup SR-NMF parameters to compute gFNs and pFNs

    :param dir_pnet_result: directory of the pNet result folder
    :param K: number of FNs
    :param Combine_Scan: False or True, whether to combine multiple scans for the same subject
    :param file_gFN: directory of a precomputed gFN in .mat format

    # model parameters
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

    # computation resource settings
    :param Parallel: False or True, whether to enable parallel computation
    :param Computation_Mode: 'CPU'
    :param N_Thread: positive integers, used for parallel computation
    :param dataPrecision: 'double' or 'single'
    :param outputFormat: 'MAT', 'Both', 'MAT' is to save results in FN.mat and TC.mat for functional networks and time courses respectively. 'Both' is for both matlab format and fMRI input file format

    :return: setting: a structure

    Yuncong Ma, 2/5/2024
    """

    dir_pnet_dataInput, dir_pnet_FNC, _, _, _, _ = setup_result_folder(dir_pnet_result)

    # Set sampleSize if it is 'Automatic'
    if sampleSize == 'Automatic':
        file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
        list_subject_ID = np.array([line.replace('\n', '') for line in open(file_subject_ID, 'r')])
        subject_ID_unique = np.unique(list_subject_ID)
        N_Subject = subject_ID_unique.shape[0]
        if sampleSize == 'Automatic':
            sampleSize = int(np.maximum(100, np.round(N_Subject / 10)))  #add int() by FY on 07/26/2024
            if N_Subject < sampleSize:  # added by hm 
                sampleSize = N_Subject #- 1  #changed by Yong Fan: for sample datasets, all subjects/scans are used
                #nBS = 10   # was 5, changed by Yong Fan

    # add nTPoints on 08/01/2024
    # add init on 08/03/2024
    BootStrap = {'samplingMethod': samplingMethod, 'sampleSize': sampleSize, 'nBS': nBS, 'nTPoints': nTPoints, 'init': init}
    if isinstance(maxIter, tuple):
        gFN_maxIter = maxIter[0]
        pFN_maxIter = maxIter[1]
    else:
        gFN_maxIter = maxIter
        pFN_maxIter = maxIter
    if isinstance(minIter, tuple):
        gFN_minIter = minIter[0]
        pFN_minIter = minIter[1]
    else:
        gFN_minIter = minIter
        pFN_minIter = minIter
    Group_FN = {'file_gFN': file_gFN,
                'BootStrap': BootStrap,
                'maxIter': gFN_maxIter, 'minIter': gFN_minIter, 'error': error,
                'normW': normW, 'Alpha': Alpha, 'Beta': Beta, 'alphaS': alphaS, 'alphaL': alphaL, 'vxI': vxI,
                'ard': ard, 'eta': eta, 'nRepeat': nRepeat}
    Personalized_FN = {'maxIter': pFN_maxIter, 'minIter': pFN_minIter, 'meanFitRatio': meanFitRatio, 'error': error,
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


def update_model_parameter(dir_pnet_result: str or None, FN_model_parameter, setting=None):
    """
    Update the model parameters in setup_SR_NMF for SR-NMF

    :param dir_pnet_result:
    :param setting: obtained from setup_SR_NMF
    :param FN_model_parameter: None or a dict containing model parameters listed in setup_SR_NMF
    :return:

    Yuncong Ma, 6/5/2024
    """

    if setting is None and dir_pnet_result is None:
        raise ValueError('One of dir_pnet_result and setting need to be set with values')
    if setting is None:
        dir_pnet_dataInput, dir_pnet_FNC, _, _, _, _ = setup_result_folder(dir_pnet_result)
        setting = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))

    # check
    if FN_model_parameter is None:
        return setting
    elif not isinstance(FN_model_parameter, dict):
        raise ValueError('FN_model_parameter needs to be either None or a dict')

    # default model parameters
    FN_Model = dict(
        file_gFN=None,
        init='random',
        samplingMethod='Subject',
        sampleSize='Automatic',
        nBS=50,
        nTPoints=99999,
        maxIter=(2000, 500),
        minIter=200,
        meanFitRatio=0.1,
        error=1e-8,
        normW=1,
        Alpha=2,
        Beta=30,
        alphaS=0,
        alphaL=0,
        vxI=0,
        ard=0,
        eta=0,
        nRepeat=5
    )

    # changes
    for i in FN_model_parameter.keys():
        FN_Model[i] = FN_model_parameter[i]

    BootStrap = {'samplingMethod': FN_Model['samplingMethod'], 'sampleSize': FN_Model['sampleSize'], 'nBS': FN_Model['nBS'],
                 'nTPoints': FN_Model['nTPoints'], 'init': FN_Model['init']}  #add nTPoints on 08/01/2024; add init on 08/03/2024
    FN_Model['BootStrap'] = BootStrap
    maxIter = FN_Model['maxIter']
    if isinstance(maxIter, tuple):
        gFN_maxIter = maxIter[0]
        pFN_maxIter = maxIter[1]
    else:
        gFN_maxIter = maxIter
        pFN_maxIter = maxIter
    minIter = FN_Model['minIter']
    if isinstance(minIter, tuple):
        gFN_minIter = minIter[0]
        pFN_minIter = minIter[1]
    else:
        gFN_minIter = minIter
        pFN_minIter = minIter
    file_gFN = setting['Group_FN']['file_gFN']

    Group_FN = {'file_gFN': FN_Model['file_gFN'],
                'BootStrap': FN_Model['BootStrap'],
                'maxIter': gFN_maxIter, 'minIter': gFN_minIter, 'error': FN_Model['error'],
                'normW': FN_Model['normW'], 'Alpha': FN_Model['Alpha'], 'Beta': FN_Model['Beta'],
                'alphaS': FN_Model['alphaS'], 'alphaL': FN_Model['alphaL'], 'vxI': FN_Model['vxI'],
                'ard': FN_Model['ard'], 'eta': FN_Model['eta'], 'nRepeat': FN_Model['nRepeat']}
    Personalized_FN = {'maxIter': pFN_maxIter, 'minIter': pFN_minIter,
                       'meanFitRatio': FN_Model['meanFitRatio'], 'error': FN_Model['error'],
                       'normW': FN_Model['normW'], 'Alpha': FN_Model['Alpha'], 'Beta': FN_Model['Beta'],
                       'alphaS': FN_Model['alphaS'], 'alphaL': FN_Model['alphaL'],
                       'vxI': FN_Model['vxI'], 'ard': FN_Model['ard'], 'eta': FN_Model['eta']}
    setting['Group_FN'] = Group_FN
    setting['Personalized_FN'] = Personalized_FN

    write_json_setting(setting, os.path.join(dir_pnet_FNC, 'Setting.json'))
    return setting


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

def compute_gNb(Brain_Template, logFile=None):
    """
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

def normalize_data_torch(data, algorithm='vp', normalization='vmax', dataPrecision='double'):
    """
    Normalize data by algorithm and normalization settings

    :param data: data in 2D matrix [dim_time, dim_space], numpy.ndarray or torch.Tensor, recommend to use reference mode to save memory
    :param algorithm: 'z' 'gp' 'vp'
    :param normalization: 'n2' 'n1' 'rn1' 'g' 'vmax'
    :param dataPrecision: 'double' or 'single'
    :return: data

    Consistent to MATLAB function normalize_data(X, algorithm, normalization, dataPrecision)
    By Yuncong Ma, 12/12/2023
    """

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D matrix")

    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch_float)
    else:
        data = data.type(torch_float)

    if algorithm.lower() == 'z':
        # standard score for each variable
        mVec = torch.mean(data, dim=1)
        sVec = torch.maximum(torch.std(data, dim=1), torch_eps)
        data = (data - mVec[:, torch.newaxis]) / sVec[:, torch.newaxis]
    elif algorithm.lower() == 'gp':
        # remove negative value globally
        minVal = torch.min(data)
        shiftVal = torch.abs(torch.minimum(minVal, torch.tensor(0.0)))
        data += shiftVal
    elif algorithm.lower() == 'vp':
        # remove negative value voxel-wisely
        minVal = torch.min(data, dim=0, keepdim=True)[0]
        shiftVal = torch.abs(torch.minimum(minVal, torch.tensor(0.0)))
        data += shiftVal
    else:
        # do nothing
        data = data

    if normalization.lower() == 'n2':
        # l2 normalization for each observation
        l2norm = torch.sqrt(torch.sum(data ** 2, dim=1)) + torch_eps
        data = data / l2norm[:, torch.newaxis]
    elif normalization.lower() == 'n1':
        # l1 normalization for each observation
        l1norm = torch.sum(data, dim=1) + torch_eps
        data = data / l1norm[:, torch.newaxis]
    elif normalization.lower() == 'rn1':
        # l1 normalization for each variable
        l1norm = torch.sum(data, dim=0) + torch_eps
        data = data / l1norm
    elif normalization.lower() == 'g':
        # global scale
        sVal = torch.sort(data, dim=None)
        perT = 0.001
        minVal = sVal[int(len(sVal) * perT)]
        maxVal = sVal[int(len(sVal) * (1 - perT))]
        data[data < minVal] = minVal
        data[data > maxVal] = maxVal
        data = (data - minVal) / max((maxVal - minVal), torch_eps)
    elif normalization.lower() == 'vmax':
        cmin = torch.min(data, dim=0, keepdim=True).values
        cmax = torch.max(data, dim=0, keepdim=True).values
        data = (data - cmin) / torch.maximum(cmax - cmin, torch_eps)
    else:
        # do nothing
        data = data

    if torch.isnan(data).any():
        raise ValueError('  nan exists, check the preprocessed data')

    return data


def initialize_u_torch(X, U0, V0, error=1e-4, maxIter=1000, minIter=30, meanFitRatio=0.1, initConv=1, dataPrecision='double'):
    """
    Initialize U with fixed V, used for pFN_NMF

    :param X: data, 2D matrix [dim_time, dim_space], numpy.ndarray or torch.Tensor
    :param U0: initial temporal component, 2D matrix [dim_time, k], numpy.ndarray or torch.Tensor
    :param V0: initial spatial component, 2D matrix [dim_space, k], numpy.ndarray or torch.Tensor
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

    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch_float)
    else:
        X = X.type(torch_float)
    if not isinstance(U0, torch.Tensor):
        U0 = torch.tensor(U0, dtype=torch_float)
    else:
        U0 = U0.type(torch_float)
    if not isinstance(X, torch.Tensor):
        V0 = torch.tensor(V0, dtype=torch_float)
    else:
        V0 = V0.type(torch_float)

    # Check the data size of X, U0 and V0
    if len(X.shape) != 2 or len(U0.shape) != 2 or len(V0.shape) != 2:
        raise ValueError("X, U0 and V0 must be 2D matrices")
    if X.shape[0] != U0.shape[0] or X.shape[1] != V0.shape[0] or U0.shape[1] != V0.shape[1]:
        raise ValueError("X, U0 and V0 need to have appropriate sizes")

    U = U0.clone()
    V = V0.clone()

    newFit = data_fitting_error_torch(X, U, V, 0, 1, dataPrecision)
    meanFit = newFit / meanFitRatio

    maxErr = 1
    for i in range(1, maxIter+1):
        # update U with fixed V
        XV = X @ V
        VV = V.T @ V
        UVV = U @ VV

        U = U * (XV / torch.maximum(UVV, torch_eps))

        if i > minIter:
            if initConv:
                newFit = data_fitting_error_torch(X, U, V, 0, 1, dataPrecision)
                meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newFit
                maxErr = (meanFit - newFit) / meanFit

        if maxErr <= error:
            break

    U_final = U
    return U_final

def data_fitting_error_torch(X, U, V, deltaVU=0, dVordU=1, dataPrecision='double'):
    """
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
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch_float)
    else:
        X = X.type(torch_float)
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=torch_float)
    else:
        U = U.type(torch_float)
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch_float)
    else:
        V = V.type(torch_float)

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
        obj_NMF = torch.sum(torch.pow(dX, 2))
        if deltaVU:
            if dVordU:
                dV = dX.T * U
            else:
                dV = dX * V
    else:
        obj_NMF = 0
        if deltaVU:
            if dVordU:
                dV = torch.zeros_like(V)
            else:
                dV = torch.zeros_like(U)
        for i in range(int(np.ceil(dim_space/nBlock))):
            if i == int(np.ceil(dim_space/nBlock)):
                smpIdx = range(i*nBlock, dim_space)
            else:
                smpIdx = range(i*nBlock, np.minimum(dim_space, (i+1)*nBlock))
            dX = (U @ V[smpIdx, :].T) - X[:, smpIdx]
            obj_NMF += torch.sum(torch.sum(torch.pow(dX, 2)))
            if deltaVU:
                if dVordU:
                    dV[smpIdx, :] = torch.dot(dX.T, U)
                else:
                    dV += torch.dot(dX, V[smpIdx, :])
        if deltaVU:
            if dVordU:
                dV = dV

    Fitting_Error = obj_NMF
    return Fitting_Error

def normalize_u_v_torch(U, V, NormV, Norm, dataPrecision='double'):
    """
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
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=torch_float)
    else:
        U = U.type(torch_float)
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch_float)
    else:
        V = V.type(torch_float)

    # Check data size of U and V
    if len(U.shape) != 2 or len(V.shape) != 2:
        raise ValueError("U and V must be 2D matrices")
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V need to have appropriate sizes")

    dim_space = V.shape[0]
    dim_time = U.shape[0]

    if Norm == 2:
        norms = torch.sqrt(torch.sum(torch.pow(V, 2), dim=0))
        norms = torch.maximum(norms, torch_eps)
    else:
        norms = torch.max(V, dim=0)[0]  # torch.max return Value and Index
        norms = torch.maximum(norms, torch_eps)

    if NormV:
        U = U * torch.tile(norms, (dim_time, 1))
        V = V / torch.tile(norms, (dim_space, 1))
    else:
        U = U / torch.tile(norms, (dim_time, 1))
        V = V * torch.tile(norms, (dim_space, 1))

    return U, V


def construct_Laplacian_gNb_torch(gNb, dim_space, vxI=0, X=None, alphaL=10, normW=1, dataPrecision='double'):
    """
    Construct Laplacian matrices for Laplacian spatial regularization term

    :param gNb: graph neighborhood, a 2D matrix [N, 2] storing rows and columns of non-zero elements
    :param dim_space: dimension of space (number of voxels or vertices)
    :param vxI: 0 or 1, flag for using the temporal correlation between nodes (vertex, voxel)
    :param X: fMRI data, a 2D matrix, [dim_time, dim_space]
    :param alphaL: internal hyper parameter for Laplacian regularization term
    :param normW: 1 or 2, normalization method for Laplacian matrix W
    :param dataPrecision: 'double' or 'single'
    :return: L, W, D: sparse 2D matrices [dim_space, dim_space]

    Yuncong Ma, 9/7/2023
    """

    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    np_float, np_eps = set_data_precision(dataPrecision)

    # Use numpy version to do
    # Current torch version does NOT support sparse matrix multiplication without CUDA
    if not isinstance(X, np.ndarray):
        X2 = np.array(X, dtype=np_float)
    else:
        X2 = X.astype(np_float)
    L, W, D = construct_Laplacian_gNb(gNb, dim_space, vxI, X2, alphaL, normW, dataPrecision)

    L = L.tocoo()
    # Create PyTorch sparse tensor using the COO format data
    indices = torch.tensor(np.array([L.row, L.col]), dtype=torch.long)
    values = torch.tensor(L.data, dtype=torch_float)
    L = torch.sparse_coo_tensor(indices, values, L.shape)

    D = D.tocoo()
    # Create PyTorch sparse tensor using the COO format data
    indices = torch.tensor(np.array([D.row, D.col]), dtype=torch.long)
    values = torch.tensor(D.data, dtype=torch_float)
    D = torch.sparse_coo_tensor(indices, values, D.shape)

    W = W.tocoo()
    # Create PyTorch sparse tensor using the COO format data
    indices = torch.tensor(np.array([W.row, W.col]), dtype=torch.long)
    values = torch.tensor(W.data, dtype=torch_float)
    W = torch.sparse_coo_tensor(indices, values, W.shape)

    return L, W, D

def robust_normalize_V(V, factor = 0.95, dataPrecision='double'):
    # Setup data precision and eps
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    # assume V is the FNs with its first dim as spatial one
    #robust normalization of V
    sdim, fdim = V.shape
    if sdim < fdim:
        print('\n  the input V has wrong dimension.' + str(sdim) + str(fdim) + '\n') 
    vtop = torch.quantile(V, 1.0-factor, dim=0, keepdim=True)
    vbottome = torch.quantile(V, factor, dim=0, keepdim=True)
    vdiff = torch.maximum(vtop-vbottome, torch_eps)
    #print(vdiff)
    #print(torch.tile(vdiff, (V.shape[0], 1)).shape)
    V = torch.clamp( (V - torch.tile(vbottome, (V.shape[0],1))) / torch.tile(vdiff, (V.shape[0], 1)), min = 0.0, max = 1.0)
    return V

def pFN_SR_NMF_torch(Data, gFN, gNb, maxIter=1000, minIter=30, meanFitRatio=0.1, error=1e-4, normW=1,
                     Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, initConv=1, ard=0, eta=0, dataPrecision='double', logFile='Log_pFN_NMF.log'):
    """
    Compute personalized FNs by spatially-regularized NMF method with group FNs as initialization

    :param Data: 2D matrix [dim_time, dim_space], numpy.ndarray or torch.Tensor. Data will be formatted to Tensor and normalized.
    :param gFN: group level FNs 2D matrix [dim_space, K], K is the number of functional networks, numpy.ndarray or torch.Tensor. gFN will be cloned
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
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)

    # Transform data format if necessary
    if not isinstance(Data, torch.Tensor):
        Data = torch.tensor(Data, dtype=torch_float)
    else:
        Data = Data.type(torch_float)
    if not isinstance(gFN, torch.Tensor):
        gFN = torch.tensor(gFN, dtype=torch_float)
    else:
        gFN = gFN.type(torch_float)
    if not isinstance(error, torch.Tensor):
        error = torch.tensor(error, dtype=torch_float)
    else:
        error = error.type(torch_float)
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, dtype=torch_float)
    else:
        eta = eta.type(torch_float)
    if not isinstance(alphaS, torch.Tensor):
        alphaS = torch.tensor(alphaS, dtype=torch_float)
    else:
        alphaS = alphaS.type(torch_float)

    # check dimension of Data and gFN
    if Data.shape[1] != gFN.shape[0]:
        raise ValueError("The second dimension of Data should match the first dimension of gFn, as they are space dimension")

    K = gFN.shape[1]

    # setup log file
    if isinstance(logFile, str):
        logFile = open(logFile, 'a')
    print(f'\nStart NMF for pFN using PyTorch at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # initialization
    initV = gFN.clone()

    dim_time, dim_space = Data.shape

    # Median number of graph neighbors
    nM = np.median(np.unique(gNb[:, 0], return_counts=True)[1])

    # Use Alpha and Beta to set alphaS and alphaL if they are 0
    if alphaS == 0 and Alpha > 0:
        alphaS = torch.tensor(np.round(Alpha * dim_time / K))
    if alphaL == 0 and Beta > 0:
        alphaL = np.round(Beta * dim_time / K / nM)

    # Prepare and normalize scan
    Data = normalize_data_torch(Data, 'vp', 'vmax', dataPrecision=dataPrecision)
    X = Data    # Save memory

    # Construct the spatial affinity graph
    L, W, D = construct_Laplacian_gNb_torch(gNb, dim_space, vxI, X, alphaL, normW, dataPrecision=dataPrecision)

    # Initialize V
    V = initV.clone()
    miv = torch.max(V, dim=0)[0]
    trimInd = V / torch.maximum(torch.tile(miv, (dim_space, 1)), torch_eps) < torch.tensor(5e-2)
    V[trimInd] = 0
    #robust normalization of V
    #V = robust_normalize_V(V, factor = 1.0/K) # try to keep 1/K non-zero values for each component, updated on 08/03/2024
    
    # Initialize U
    U = X @ V / torch.tile(torch.sum(V, dim=0), (dim_time, 1))

    U = initialize_u_torch(X, U, V, error=error, maxIter=100, minIter=30, meanFitRatio=meanFitRatio, initConv=initConv, dataPrecision=dataPrecision)

    initV = V.clone()

    # Alternative update of U and V
    # Variables

    if ard > 0:
        lambdas = torch.sum(U, dim=0) / dim_time
        hyperLam = eta * torch.sum(torch.pow(X, 2)) / (dim_time * dim_space * 2)
    else:
        lambdas = 0
        hyperLam = 0

    flagQC = 0
    oldLogL = torch.inf
    oldU = U.clone()
    oldV = V.clone()

    for i in range(1, 1+maxIter):
        # ===================== update V ========================
        # Eq. 8-11
        XU = X.T @ U
        UU = U.T @ U
        VUU = V @ UU

        tmpl2 = torch.pow(V, 2)

        if alphaS > 0:
            tmpl21 = torch.sqrt(tmpl2)
            tmpl22 = torch.tile(torch.sqrt(torch.sum(tmpl2, dim=0)), (dim_space, 1))
            tmpl21s = torch.tile(torch.sum(tmpl21, dim=0), (dim_space, 1))
            posTerm = V / torch.maximum(tmpl21 * tmpl22, torch_eps)
            negTerm = V * tmpl21s / torch.maximum(torch.pow(tmpl22, 3), torch_eps)

            VUU = VUU + 0.5 * alphaS * posTerm
            XU = XU + 0.5 * alphaS * negTerm

        if alphaL > 0:
            WV = W @ V
            DV = D @ V

            XU = XU + WV
            VUU = VUU + DV

        V = V * (XU / torch.maximum(VUU, torch_eps))

        # Prune V if empty components are found in V
        # This is almost impossible to happen without combining FNs
        # prunInd = torch.sum(V != 0, dim=0) == 1
        # if torch.any(prunInd):
        #     V[:, prunInd] = torch.zeros((dim_space, torch.sum(prunInd)), dtype=torch_float)
        #     U[:, prunInd] = torch.zeros((dim_time, torch.sum(prunInd)), dtype=torch_float)

        # normalize U and V
        U, V = normalize_u_v_torch(U, V, 1, 1, dataPrecision=dataPrecision)

        # ===================== update U =========================
        XV = X @ V
        VV = V.T @ V
        UVV = U @ VV

        if ard > 0:  # ard term for U
            posTerm = torch.tensor(1) / torch.maximum(torch.tile(lambdas, (dim_time, 1)), torch_eps)

            UVV = UVV + posTerm * hyperLam

        U = U * (XV / torch.maximum(UVV, torch_eps))

        # Prune U if empty components are found in U
        # This is almost impossible to happen without combining FNs
        # prunInd = torch.sum(U, dim=0) == 0
        # if torch.any(prunInd):
        #     V[:, prunInd] = torch.zeros((dim_space, torch.sum(prunInd)), dtype=torch_float)
        #     U[:, prunInd] = torch.zeros((dim_time, torch.sum(prunInd)), dtype=torch_float)

        # update lambda
        if ard > 0:
            lambdas = torch.sum(U, dim=0) / dim_time

        # ==== calculate objective function value ====
        ardU = 0
        tmp1 = 0
        tmp2 = 0
        tmp3 = 0
        tmpl21 = torch.pow(V, 2)

        if ard > 0:
            su = torch.sum(U, dim=0)
            su[su == 0] = 1
            ardU = torch.sum(torch.log(su)) * dim_time * hyperLam

        tmpDf = torch.pow(X - U @ V.T, 2)
        tmp1 = torch.sum(tmpDf)

        if alphaL > 0:
            tmp2 = V.T @ L * V.T

        L21 = alphaS * torch.sum(torch.sum(torch.sqrt(tmpl21), dim=0) / torch.maximum(torch.sqrt(torch.sum(tmpl21, dim=0)), torch_eps))
        LDf = tmp1
        LSl = torch.sum(tmp2)

        # Objective function
        LogL = L21 + ardU + LDf + LSl
        print(f"    Iter = {i}: LogL: {LogL}, dataFit: {LDf}, spaLap: {LSl}, L21: {L21}, ardU: {ardU}", file=logFile)

        # The iteration needs to meet minimum iteration number and small changes of LogL
        if i > minIter and abs(oldLogL - LogL) / torch.maximum(oldLogL, torch_eps) < error:
            break
        oldLogL = LogL.clone()

        # QC Control
        temp = mat_corr_torch(gFN, V, dataPrecision=dataPrecision)
        QC_Spatial_Correspondence = torch.clone(torch.diag(temp))
        temp -= torch.diag(2 * torch.ones(K))  # set diagonal values to lower than -1
        QC_Spatial_Correspondence_Control = torch.max(temp, dim=0)[0]
        QC_Delta_Sim = torch.min(QC_Spatial_Correspondence - QC_Spatial_Correspondence_Control)
        QC_Delta_Sim = QC_Delta_Sim.cpu().numpy()

        if QC_Delta_Sim <= 0:
            flagQC = 1
            U = oldU.clone()
            V = oldV.clone()
            print(f'\n  QC: Meet QC constraint: Delta sim = {QC_Delta_Sim}', file=logFile, flush=True)
            print(f'    Use results from last iteration', file=logFile, flush=True)
            break
        else:
            oldU = U.clone()
            oldV = V.clone()
            print(f'        QC: Delta sim = {QC_Delta_Sim}', file=logFile, flush=True)

    print(f'\n Finished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    return U, V


def gFN_SR_NMF_torch(Data, K, gNb, init='random', maxIter=1000, minIter=200, error=1e-8, normW=1,
                     Alpha=2, Beta=30, alphaS=0, alphaL=0, vxI=0, ard=0, eta=0, nRepeat=5, dataPrecision='double', logFile='Log_gFN_NMF.log'):
    """
    Compute group-level FNs using NMF method

    :param Data: 2D matrix [dim_time, dim_space], numpy.ndarray or torch.Tensor, recommend to normalize each fMRI scan before concatenate them along the time dimension
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

    Yuncong Ma, 2/2/2024
    """

    # setup log file
    if isinstance(logFile, str):
        logFile = open(logFile, 'a')
    print(f'\nStart NMF for gFN using PyTorch at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # Setup data precision and eps
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)

    # Transform data format if necessary
    if not isinstance(Data, torch.Tensor):
        Data = torch.tensor(Data, dtype=torch_float)
    else:
        Data = Data.type(torch_float)
    if not isinstance(error, torch.Tensor):
        error = torch.tensor(error, dtype=torch_float)
    else:
        error = error.type(torch_float)
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, dtype=torch_float)
    else:
        eta = eta.type(torch_float)
    if not isinstance(alphaS, torch.Tensor):
        alphaS = torch.tensor(alphaS, dtype=torch_float)
    else:
        alphaS = alphaS.type(torch_float)

    # Input data size
    dim_time, dim_space = Data.shape

    # Median number of graph neighbors
    nM = np.median(np.unique(gNb[:, 0], return_counts=True)[1])

    # Use Alpha and Beta to set alphaS and alphaL if they are 0
    if alphaS == 0 and Alpha > 0:
        alphaS = torch.tensor(np.round(Alpha * dim_time / K))
    if alphaL == 0 and Beta > 0:
        alphaL = np.round(Beta * dim_time / K / nM)

    # Prepare and normalize scan
    Data = normalize_data_torch(Data, 'vp', 'vmax', dataPrecision=dataPrecision)
    X = Data  # Save memory

    # Construct the spatial affinity graph
    L, W, D = construct_Laplacian_gNb_torch(gNb, dim_space, vxI, X, alphaL, normW, dataPrecision)

    '''
    #add initialization parameter on 08/03/2024
    from sklearn.decomposition import NMF   # for sklearn based NMF initialization
    model = NMF(n_components=K, init=init, max_iter=1, solver='mu')
    model._check_params(X.numpy())
    U = None
    V = None
    U, V = model._check_w_h(X.numpy(), U, V,True)
    U = torch.tensor(U)
    V = torch.tensor(V).T
    '''
    if init != 'random':
        # to use sklearn NMF  on Aug 07, 2024
        from sklearn.decomposition import NMF
        model = NMF(n_components=K, init=init, max_iter=20000) #, random_state=0)
        W = model.fit_transform(X.cpu().numpy())
        H = model.components_
        return torch.tensor(np.transpose(H))

    flag_Repeat = 0
    for repeat in range(1, 1 + nRepeat):
        flag_Repeat = 0
        print(f'\n Starting {repeat}-th repetition\n', file=logFile, flush=True)

        # Initialize U and V
        mean_X = torch.divide(torch.sum(X), torch.tensor(dim_time*dim_space))
        U = (torch.rand((dim_time, K), dtype=torch_float) + 1) * (torch.sqrt(torch.div(mean_X, K)))
        V = (torch.rand((dim_space, K), dtype=torch_float) + 1) * (torch.sqrt(torch.div(mean_X, K)))
                
        # Normalize data
        U, V = normalize_u_v_torch(U, V, 1, 1, dataPrecision)

        #robust normalization of V
        #V = robust_normalize_V(V, factor = 1.0/K) # try to keep 1/K non-zero values for each component, updated on 08/03/2024

        if ard > 0:
            ard = 1
            eta = 0.1
            lambdas = torch.sum(U, dim=0) / dim_time
            hyperLam = eta * torch.sum(torch.pow(X, 2)) / (dim_time * dim_space * 2)
        else:
            lambdas = 0
            hyperLam = 0

        oldLogL = torch.inf

        # Multiplicative update of U and V
        for i in range(1, 1+maxIter):
            # ===================== update V ========================
            # Eq. 8-11
            XU = torch.matmul(X.T, U)
            UU = torch.matmul(U.T, U)
            VUU = torch.matmul(V, UU)

            tmpl2 = torch.pow(V, 2)

            if alphaS > 0:
                tmpl21 = torch.sqrt(tmpl2)
                tmpl22 = torch.sqrt(torch.sum(tmpl2, dim=0, keepdim=True))  # tmpl22 = torch.tile(torch.sqrt(torch.sum(tmpl2, dim=0)), (dim_space, 1))
                tmpl21s = torch.sum(tmpl21, dim=0, keepdim=True)  # tmpl21s = torch.tile(torch.sum(tmpl21, dim=0), (dim_space, 1))
                posTerm = torch.div(V, torch.maximum(torch.mul(tmpl21, tmpl22), torch_eps))
                negTerm = torch.div(torch.mul(V, tmpl21s), torch.maximum(torch.pow(tmpl22, 3), torch_eps))

                VUU = VUU + 0.5 * alphaS * posTerm
                XU = XU + 0.5 * alphaS * negTerm

            if alphaL > 0:
                WV = torch.matmul(W, V)  # WV = torch.matmul(W, V.type(torch.float64))
                DV = torch.matmul(D, V)  # DV = torch.matmul(D, V.type(torch.float64))

                XU = torch.add(XU, WV)
                VUU = torch.add(VUU, DV)

            V = torch.mul(V, (torch.div(XU, torch.maximum(VUU, torch_eps))))

            # Prune V if empty components are found in V
            # This is almost impossible to happen without combining FNs
            prunInd = torch.sum(V != 0, dim=0) == 1
            if torch.any(prunInd):
                V[:, prunInd] = torch.zeros((dim_space, torch.sum(prunInd)))
                U[:, prunInd] = torch.zeros((dim_time, torch.sum(prunInd)))

            # normalize U and V
            U, V = normalize_u_v_torch(U, V, 1, 1, dataPrecision)

            # ===================== update U =========================
            XV = torch.matmul(X, V)
            VV = torch.matmul(V.T, V)
            UVV = torch.matmul(U, VV)

            if ard > 0:  # ard term for U
                posTerm = torch.div(torch.tensor(1.0), torch.maximum(torch.tile(lambdas, (dim_time, 1)), torch_eps))

                UVV = torch.add(UVV, posTerm * hyperLam)

            U = torch.mul(U, torch.div(XV, torch.maximum(UVV, torch_eps)))

            # Prune U if empty components are found in U
            # This is almost impossible to happen without combining FNs
            prunInd = torch.sum(U, dim=0) == 0
            if torch.any(prunInd):
                V[:, prunInd] = torch.zeros((dim_space, torch.sum(prunInd)))
                U[:, prunInd] = torch.zeros((dim_time, torch.sum(prunInd)))

            # update lambda
            if ard > 0:
                lambdas = torch.sum(U) / dim_time

            # ==== calculate objective function value ====
            ardU = 0
            tmp2 = 0
            tmpl21 = torch.pow(V, 2)

            if ard > 0:
                su = torch.sum(U, dim=0)
                su[su == 0] = 1
                ardU = torch.sum(torch.log(su)) * dim_time * hyperLam

            if alphaL > 0:
                tmp2 = torch.mul(torch.matmul(V.T, L), V.T)

            L21 = torch.mul(alphaS, torch.sum(torch.div(torch.sum(torch.sqrt(tmpl21), dim=0), torch.maximum(torch.sqrt(torch.sum(tmpl21, dim=0)), torch_eps))))
            # LDf = data_fitting_error(X, U, V, 0, 1)
            LDf = torch.sum(torch.pow(torch.subtract(X, torch.matmul(U, V.T)), 2))
            LSl = torch.sum(tmp2)

            # Objective function
            LogL = L21 + LDf + LSl + ardU
            print(f"    Iter = {i}: LogL: {LogL}, dataFit: {LDf}, spaLap: {LSl}, L21: {L21}, ardU: {ardU}", file=logFile, flush=True)

            if 1 < i < minIter and abs(oldLogL - LogL) / torch.maximum(oldLogL, torch_eps) < error:
                flag_Repeat = 1
                print('\n Iteration stopped before the minimum iteration number. The results might be poor.\n', file=logFile, flush=True)
                break
            elif i > minIter and abs(oldLogL - LogL) / torch.maximum(oldLogL, torch_eps) < error:
                break
            oldLogL = LogL.clone()
        if flag_Repeat == 0:
            break

    if flag_Repeat == 1:
        print('\n All repetition stopped before the minimum iteration number. The final results might be poor\n', file=logFile, flush=True)

    gFN = V
    print(f'\nFinished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)
    return gFN


def gFN_fusion_NCut_torch(gFN_BS, K, NCut_MaxTrial=100, dataPrecision='double', logFile='Log_gFN_fusion_NCut'):
    """
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
    print(f'\nStart NCut for gFN fusion using PyTorch at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    # Setup data precision and eps
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)

    if not isinstance(gFN_BS, torch.Tensor):
        gFN_BS = torch.tensor(gFN_BS, dtype=torch_float)
    else:
        gFN_BS = gFN_BS.type(torch_float)

    # clustering by NCut

    # Get similarity between samples
    corrVal = mat_corr_torch(gFN_BS, dataPrecision=dataPrecision)  # similarity between FNs, [K * n_BS, K * n_BS]
    corrVal[torch.isnan(corrVal)] = -1
    nDis = 1 - corrVal  # Transform Pearson correlation to non-negative values similar to distance
    triuInd = torch.triu(torch.ones(nDis.shape), 1)  # Index of upper triangle
    nDisVec = nDis[triuInd == 1]  # Take the values in upper triangle
    # Make all distance non-negative and normalize their distribution
    nW = torch.exp(-torch.pow(nDis, 2) / (torch.pow(torch.median(nDisVec), 2)))  # Transform distance values using exp(-X/std^2) with std as the median value
    nW[torch.isnan(nW)] = 0
    sumW = torch.sum(nW, dim=1)  # total distance for each FN
    sumW[sumW == 0] = 1  # In case two FNs are the same
    # Construct Laplacian matrix
    D = torch.diag(sumW)
    L = torch.sqrt(torch.linalg.inv(D)) @ nW @ torch.linalg.inv(torch.sqrt(D))  # A way to normalize nW based on the total distance of each FN
    L = (L + L.T) / 2  # Ensure L is symmetric. Computation error may result in asymmetry

    # Get first K eigenvectors, sign of vectors may be different to MATLAB results
    eigenvalues, eigenvectors = torch.linalg.eigh(L.type(torch_float))
    # Sort by eigenvalues and select the K largest
    sorted_indices = torch.argsort(eigenvalues, descending=True)[:K]
    eVal = eigenvalues[sorted_indices]
    Ev = eigenvectors[:, sorted_indices]
    Ev = torch.real(Ev)
    # Correct the sign of eigenvectors to make them same as derived from MATLAB
    temp = torch.sign(torch.sum(Ev, dim=0))  # Use the total value of each eigenvector to reset its sign
    temp[temp == 0.0] = 1.0
    Ev = Ev * torch.tile(temp, (Ev.shape[0], 1))  # Reset the sign of each eigenvector
    normvect = torch.sqrt(torch.diag(Ev @ Ev.T))  # Get the norm of each eigenvector
    normvect[normvect == 0.0] = 1  # Incase all 0 eigenvector
    Ev = torch.linalg.solve(torch.diag(normvect), Ev)  # Use linear solution to normalize Ev satisfying normvect * Ev_new = Ev_old

    # Multiple trials to get reproducible results
    Best_C = []
    Best_NCutValue = torch.inf
    for i in range(1,NCut_MaxTrial+1):
        print(f'    Iter = ' + str(i), file=logFile)
        EigenVectors = Ev
        n, k = EigenVectors.shape  # n is K * n_BS, k is K

        vm = torch.sqrt(torch.sum(EigenVectors**2, dim=1, keepdims=True))  # norm of each row
        EigenVectors = EigenVectors / torch.tile(vm, (1, k))  # normalize eigenvectors to ensure each FN vector's norm = 1

        R = torch.zeros((k, k), dtype=torch_float)
        ps = torch.randint(0, n, (1,))  # Choose a random row in eigenvectors
        R[:, 0] = EigenVectors[ps, :]  # This randomly selected row in eigenvectors is used as an initial center

        c_index = torch.zeros(n)
        c = torch.zeros(n, dtype=torch_float)  # Total distance to different rows in R [K * n_BS, 1]
        c_index[0] = ps  # Store the index of selected samples
        c[ps] = torch.inf

        for j in range(2, k+1):  # Find another K-1 rows in eigenvectors which have the minimum similarity to previous selected rows, similar to initialization in k++
            c += torch.abs(EigenVectors @ R[:, j-2])
            ps = torch.argmin(c)
            c_index[j-1] = ps
            c[ps] = torch.inf
            R[:, j-1] = EigenVectors[ps, :]

        lastObjectiveValue = 0
        exitLoop = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        while exitLoop == 0:
            nbIterationsDiscretisation += 1

            EigenVectorsR = EigenVectors @ R
            n, k = EigenVectorsR.shape
            J = torch.argmax(EigenVectorsR, dim=1)  # Assign each sample to K centers of R based on highest similarity
            Indice = torch.stack((torch.arange(n).type(torch.int32), J.type(torch.int32)))
            EigenvectorsDiscrete = torch.sparse_coo_tensor(Indice, torch.ones(n, dtype=torch_float), (n, k))  # Generate a 0-1 matrix with each row containing only one 1

            U, S, Vh = torch.linalg.svd(EigenvectorsDiscrete.T @ EigenVectors, full_matrices=False)  # Economy-size decomposition

            S = torch.diag(S)  # To match MATLAB svd results
            V = Vh.T  # To match MATLAB svd results
            NcutValue = 2 * (n - torch.trace(S))

            # escape the loop when converged or meet max iteration
            if abs(NcutValue - lastObjectiveValue) < torch_eps or nbIterationsDiscretisation > nbIterationsDiscretisationMax:
                exitLoop = 1
                print(f'    Reach stop criterion of NCut, NcutValue = '+str(NcutValue.numpy())+'\n', file=logFile, flush=True)
            else:
                print(f'    NcutValue = '+str(NcutValue.numpy()), file=logFile)
                lastObjectiveValue = NcutValue
                R = V @ U.T  # Update R which stores the new centers

        C = torch.argmax(EigenvectorsDiscrete.to_dense(), dim=1)  # Assign each sample to K centers in R

        if len(torch.unique(C)) < K:  # Check whether there are empty results
            print(f'    Found empty results in iteration '+str(i)+'\n', file=logFile, flush=True)
        else:  # Update the best result
            if NcutValue < Best_NCutValue:
                Best_NCutValue = NcutValue
                Best_C = C

    if len(set(Best_C)) < K:  # In case even the last trial has empty results
        raise ValueError('  Cannot generate non-empty gFNs\n')

    print(f'Best NCut value = '+str(Best_NCutValue.numpy())+'\n', file=logFile, flush=True)

    # Get centroid
    C = Best_C
    gFN = torch.zeros((gFN_BS.shape[0], K))
    for ki in range(K):
        if torch.sum(C == ki) > 1:
            candSet = gFN_BS[:, C == ki]  # Get the candidate set of FNs assigned to cluster ki
            corrW = torch.abs(mat_corr_torch(candSet, dataPrecision=dataPrecision))  # Get the similarity between candidate FNs
            corrW[torch.isnan(corrW)] = 0
            mInd = torch.argmax(torch.sum(corrW, dim=0), dim=0)  # Find the FN with the highest total similarity to all other FNs
            gFN[:, ki] = candSet[:, mInd]
        elif torch.sum(C == ki) == 1:
            mInd = int((C == ki).nonzero(as_tuple=True)[0])
            gFN[:, ki] = gFN_BS[:, mInd]

    gFN = gFN / torch.maximum(torch.tile(torch.max(gFN, dim=0)[0], (gFN.shape[0], 1)), torch_eps)  # Normalize each FN by its max value
    print(f'\nFinished at '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n', file=logFile, flush=True)

    return gFN

def gFN_SR_NMF_boostrapping_cluster(dir_pnet_result: str, jobID=1):
    """
    Run the NMF for bootstraping in cluster computation

    :param dir_pnet_result: directory of pNet result folder
    :param jobID: jobID starting from 1
    :return: None

    Yuncong Ma, 2/2/2024
    """

    jobID = int(jobID)

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # get settings
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC}

    # load basic settings
    dataType = setting['Data_Input']['Data_Type']
    dataFormat = setting['Data_Input']['Data_Format']

    # load Brain Template
    Brain_Template = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))
    if dataType == 'Volume':
        Brain_Mask = Brain_Template['Brain_Mask']
    else:
        Brain_Mask = None

    # Extract parameters
    K = setting['FN_Computation']['K']
    init = setting['FN_Computation']['Group_FN']['BootStrap']['init']
    nTPoints = setting['FN_Computation']['Group_FN']['BootStrap']['nTPoints'] 
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
    if isinstance(maxIter, int) or (isinstance(maxIter, np.ndarray) and maxIter.shape == 1):
        maxIter_gFN = maxIter
        maxIter_pFN = maxIter
    else:
        maxIter_gFN = maxIter[0]
        maxIter_pFN = maxIter[1]
    if isinstance(minIter, int) or (isinstance(minIter, np.ndarray) and minIter.shape == 1):
        minIter_gFN = minIter
        minIter_pFN = minIter
    else:
        minIter_gFN = minIter[0]
        minIter_pFN = minIter[1]

    dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
    if not os.path.exists(dir_pnet_BS):
        os.makedirs(dir_pnet_BS)

    # NMF on bootstrapped subsets
    print('Start to SR-NMF for this bootstrap at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
    dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')

    # load data
    file_scan_list = os.path.join(dir_pnet_BS, str(jobID), 'Scan_List.txt')
    Data, CHeader, NHeader = load_fmri_scan(file_scan_list, dataType=dataType, dataFormat=dataFormat, nTPoints=nTPoints, Reshape=True, Brain_Mask=Brain_Mask,
                          Normalization='vp-vmax', logFile=None)

    # additional parameter
    gNb = load_matlab_single_array(os.path.join(dir_pnet_FNC, 'gNb.mat'))

    # perform NMF
    FN_BS = gFN_SR_NMF_torch(Data, K, gNb, maxIter=maxIter_gFN, init=init, minIter=minIter_gFN, error=error, normW=normW,
                                    Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL, vxI=vxI, ard=ard, eta=eta,
                                    nRepeat=nRepeat, dataPrecision=dataPrecision, logFile=None)
    # save results
    FN_BS = reshape_FN(FN_BS.numpy(), dataType=dataType, Brain_Mask=Brain_Mask)
    sio.savemat(os.path.join(dir_pnet_BS, str(jobID), 'FN.mat'), {"FN": FN_BS}, do_compression=True)
    # save FNs in nii.gz and TC as txt file  FY 07/26/2024
    output_FN(FN=FN_BS,
              file_output=os.path.join(dir_pnet_BS, str(jobID), 'FN.mat'),
              file_brain_template = Brain_Template,
              dataFormat=dataFormat, 
              Cheader = CHeader,
              Nheader = NHeader)
    print('Finished at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)

def fuse_FN_cluster(dir_pnet_result: str):
    """
    Run the NCut to fuse FNs in cluster computation

    :param dir_pnet_result: directory of pNet result folder
    :return: None

    Yuncong Ma, 2/2/2024
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # get settings
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC}

    # load basic settings
    dataType = setting['Data_Input']['Data_Type']
    dataFormat = setting['Data_Input']['Data_Format']

    # load Brain Template
    Brain_Template = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))
    if dataType == 'Volume':
        Brain_Mask = Brain_Template['Brain_Mask']
    else:
        Brain_Mask = None
    print('Brain template is loaded from folder Data_Input', flush=True)

    print('Start to fuse bootstrapped results using NCut at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
    dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
    K = setting['FN_Computation']['K']
    nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']
    FN_BS = np.empty(nBS, dtype=np.ndarray)
    # load bootstrapped results
    for rep in range(1, nBS+1):
        FN_BS[rep-1] = np.array(reshape_FN(load_matlab_single_array(os.path.join(dir_pnet_BS, str(rep), 'FN.mat')), dataType=dataType, Brain_Mask=Brain_Mask))
    gFN_BS = np.concatenate(FN_BS, axis=1)

    # Fuse bootstrapped results
    gFN = gFN_fusion_NCut_torch(gFN_BS, K, logFile=None)
    # output
    gFN = reshape_FN(gFN.numpy(), dataType=dataType, Brain_Mask=Brain_Mask)
    sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
    # save FNs in nii.gz and TC as txt file  FY 07/26/2024
    output_FN(FN=gFN,
              file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
              file_brain_template = Brain_Template,
              dataFormat=dataFormat)
    print('Finished at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)

def pFN_SR_NMF_cluster(dir_pnet_result: str, jobID=1):
    """
    Run the SR-NMF for pFNs in cluster computation

    :param dir_pnet_result: directory of pNet result folder
    :param jobID: jobID starting from 1
    :return: None

    Yuncong Ma, 2/2/2024
    """

    jobID = int(jobID)

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # get settings
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC}

    # load basic settings
    dataType = setting['Data_Input']['Data_Type']
    dataFormat = setting['Data_Input']['Data_Format']

    # load Brain Template
    Brain_Template = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))
    if dataType == 'Volume':
        Brain_Mask = Brain_Template['Brain_Mask']
    else:
        Brain_Mask = None

    print('Start to compute pFNs at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
    # load precomputed gFNs
    gFN = load_matlab_single_array(os.path.join(dir_pnet_gFN, 'FN.mat'))
    # additional parameter
    gNb = load_matlab_single_array(os.path.join(dir_pnet_FNC, 'gNb.mat'))
    # reshape to 2D if required
    gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
    # get subject folder in Personalized_FN
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
    list_subject_folder = [line.replace('\n', '') for line in open(file_subject_folder, 'r')]
    list_subject_folder = np.array(list_subject_folder)
    list_subject_folder_unique = np.unique(list_subject_folder)

    print(f'Start to compute pFNs for {jobID}-th folder: {list_subject_folder_unique[jobID-1]}', flush=True)
    dir_pnet_pFN_indv = os.path.join(dir_pnet_pFN, list_subject_folder_unique[jobID-1])
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
    if isinstance(maxIter, int) or (isinstance(maxIter, np.ndarray) and maxIter.shape == 1):
        maxIter_gFN = maxIter
        maxIter_pFN = maxIter
    else:
        maxIter_gFN = maxIter[0]
        maxIter_pFN = maxIter[1]
    if isinstance(minIter, int) or (isinstance(minIter, np.ndarray) and minIter.shape == 1):
        minIter_gFN = minIter
        minIter_pFN = minIter
    else:
        minIter_gFN = minIter[0]
        minIter_pFN = minIter[1]

    # load data
    Data, CHeader, NHeader = load_fmri_scan(os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt'),
                          dataType=dataType, dataFormat=dataFormat,
                          Reshape=True, Brain_Mask=Brain_Mask, logFile=None)
    # perform NMF
    TC, pFN = pFN_SR_NMF_torch(Data, gFN, gNb, maxIter=maxIter_pFN, minIter=minIter_pFN, meanFitRatio=meanFitRatio,
                                      error=error, normW=normW,
                                      Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL,
                                      vxI=vxI, ard=ard, eta=eta,
                                      dataPrecision=dataPrecision, logFile=None)
    pFN = pFN.numpy()
    TC = TC.numpy()

    # output
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
    print('Finished at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
