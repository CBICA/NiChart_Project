# Yuncong Ma, 2/2/2024
# GIG-ICA method for the FN computation module of pNet
# Pytorch version
# To avoid same function naming, use import GIG_ICA

#########################################
# Packages


# other functions of pNet
from Module.Data_Input import *
from Basic.Matrix_Computation import *


def setup_GIG_ICA(dir_pnet_result: str or None, K=17, Combine_Scan=False, file_gFN=None,
                  maxIter=100, a=0.5, Nemda=1, ftol=0.02, error=1e-5,
                  EGv=0.3745672075, ErChuPai=0.6366197723,
                  Parallel=False, Computation_Mode='CPU', N_Thread=1, dataPrecision='double', outputFormat='MAT'):
    """
    Setup GIG-ICA parameters for obtaining pFNs

    :param dir_pnet_result: directory of the pNet result folder
    :param K: number of FNs
    :param Combine_Scan: False or True, whether to combine multiple scans for the same subject
    :param file_gFN: directory of a precomputed gFN in .mat format

    # model parameters
    :param maxIter: maximum iteration number for multiplicative update
    :param a: weighting to the own pFN
    :param Nemda: step size for iteration
    :param ftol: tolerance for the objective function
    :param error: error tolerance for w to obtain pFN
    :param EGv: constant
    :param ErChuPai: constant

    # computation resource settings
    :param Parallel: False or True, whether to enable parallel computation
    :param Computation_Mode: 'CPU'
    :param N_Thread: positive integers, used for parallel computation
    :param dataPrecision: 'double' or 'single'
    :param outputFormat: 'MAT', 'Both', 'MAT' is to save results in FN.mat and TC.mat for functional networks and time courses respectively. 'Both' is for both matlab format and fMRI input file format

    :return: setting: a structure

    Yuncong Ma, 2/12/2024
    """
    if dir_pnet_result is not None:
        _, dir_pnet_FNC, _, _, _, _ = setup_result_folder(dir_pnet_result)
    else:
        dir_pnet_FNC = None

    Group_FN = {'file_gFN': file_gFN}
    Personalized_FN = {'maxIter': maxIter, 'a': a, 'Nemda': Nemda, 'ftol': ftol, 'error': error,
                       'EGv': EGv, 'ErChuPai': ErChuPai}
    Computation = {'Parallel': Parallel,
                   'Model': Computation_Mode,
                   'N_Thread': N_Thread,
                   'dataPrecision': dataPrecision}

    setting = {'Method': 'GIG-ICA',
               'K': K,
               'Combine_Scan': Combine_Scan,
               'Group_FN': Group_FN,
               'Personalized_FN': Personalized_FN,
               'Computation': Computation,
               'Output_Format': outputFormat}

    if dir_pnet_FNC is not None:
        write_json_setting(setting, os.path.join(dir_pnet_FNC, 'Setting.json'))
    return setting


def update_model_parameter(dir_pnet_result: str is None, FN_model_parameter, setting=None):
    """
    Update the model parameters in setup_SR_NMF for GIG-ICA

    :param dir_pnet_result:
    :param setting: obtained from setup_GIG_ICA
    :param FN_model_parameter: None or a dict containing model parameters listed in setup_GIG_ICA
    :return:

    Yuncong Ma, 2/12/2024
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
        maxIter=100,
        a=0.5,
        Nemda=1,
        ftol=0.02,
        error=1e-5,
        EGv=0.3745672075,
        ErChuPai=0.6366197723
        )

    # changes to model parameters
    for i in FN_model_parameter.keys():
        FN_Model[i] = FN_model_parameter[i]

    # update setting
    Personalized_FN = {'maxIter': FN_Model['maxIter'], 'a': FN_Model['a'], 'Nemda': FN_Model['Nemda'],
                       'ftol': FN_Model['ftol'], 'error': FN_Model['error'],
                       'EGv': FN_Model['EGv'], 'ErChuPai': FN_Model['ErChuPai']}

    setting['Personalized_FN'] = Personalized_FN

    if dir_pnet_FNC is not None:
        write_json_setting(setting, os.path.join(dir_pnet_FNC, 'Setting.json'))
    return setting


def pFN_GIG_ICA_torch(Data, gFN, maxIter=100, a=0.5, Nemda=1, ftol=0.02, error=1e-5,
                      EGv=0.3745672075, ErChuPai=0.6366197723,
                      dataPrecision='double', logFile='Log_pFN_GIG_ICA.log'):
    """
    Compute personalized FNs using GIG-ICA

    :param Data: 2D matrix [dim_time, dim_space], numpy.ndarray or torch.Tensor. Data will be formatted to Tensor and normalized.
    :param gFN: group level FNs 2D matrix [dim_space, K], K is the number of functional networks, numpy.ndarray or torch.Tensor. gFN will be cloned
    :param maxIter: maximum iteration number for multiplicative update
    :param a: weighting to the own pFN
    :param Nemda: step size for iteration
    :param ftol: tolerance for the objective function
    :param error: error tolerance for w to obtain pFN
    :param EGv: constant
    :param ErChuPai: constant
    :param dataPrecision: 'single' or 'float32', 'double' or 'float64'
    :param logFile: str, directory of a txt log file
    :return: TC and FN. TC is the temporal components of pFNs, a 2D matrix [dim_time, K], and FN is the spatial components of pFNs, a 2D matrix [dim_space, K]

    Yuncong Ma, 2/2/2024
    """

    # setup log file
    if isinstance(logFile, str):
        logFile = open(logFile, 'a')
    print(f'\nStart GIG-ICA for pFN using PyTorch at ' +
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+'\n',
          file=logFile, flush=True)

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
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch_float)
    else:
        a = a.type(torch_float)
    if not isinstance(EGv, torch.Tensor):
        EGv = torch.tensor(EGv, dtype=torch_float)
    else:
        EGv = EGv.type(torch_float)
    if not isinstance(ErChuPai, torch.Tensor):
        ErChuPai = torch.tensor(ErChuPai, dtype=torch_float)
    else:
        ErChuPai = ErChuPai.type(torch_float)

    b = 1-a

    thr = torch.finfo(Data.dtype).eps
    n, m = Data.shape
    dim_t, dim_space = Data.shape  # [Nt Nv]
    ICRefMax = torch.t(gFN)  # [K Nv]

    FmriMat = Data - torch.mean(Data, dim=1, keepdim=True)
    CovFmri = torch.mm(FmriMat, FmriMat.t()) / m  # [Nt Nt]
    eigenvalues, eigenvectors = torch.linalg.eigh(CovFmri)
    eigenvalues = torch.real(eigenvalues)
    eigenvectors = torch.real(eigenvectors)
    # Correct the sign of eigenvectors to make them same as derived from MATLAB
    temp = torch.sign(torch.sum(eigenvectors, dim=0))  # Use the total value of each eigenvector to reset its sign
    temp[temp == 0.0] = 1.0
    eigenvectors = eigenvectors * torch.tile(temp, (eigenvectors.shape[0], 1))  # Reset the sign of each eigenvector

    # select eigenvectors with values larger than eps
    Esort, dsort = eigenvectors, eigenvalues
    filter_inds = dsort >= 0  # remove negative eigenvalues
    dsort = dsort[filter_inds]
    Esort = Esort[:, filter_inds]
    dsort = torch.abs(dsort)

    flipped_inds = torch.argsort(dsort, descending=True)
    dsort = dsort[flipped_inds]

    numpc = torch.sum(dsort > thr)  # remove small eigenvalues
    Esort = Esort[:, flipped_inds]

    K = ICRefMax.shape[0]

    Epart = Esort[:, :numpc]
    dpart = dsort[:numpc]
    Lambda_part = torch.diag(dpart)
    WhitenMatrix = torch.linalg.solve(torch.tensor(scipy.linalg.sqrtm(Lambda_part), dtype=torch_float), Epart.t())
    Y = torch.mm(WhitenMatrix, FmriMat)

    if thr < 1e-10 and numpc < dim_t:
        for i in range(Y.shape[0]):
            Y[i, :] = Y[i, :] / torch.std(Y[i, :])

    Yinv = torch.pinverse(Y)

    # normalized ICRefMax to z scores
    gFNN = torch.zeros((K, dim_space), dtype=torch_float)  # [K Nv]
    gFNC = ICRefMax - torch.mean(ICRefMax, dim=1, keepdim=True)
    for i in range(K):
        gFNN[i, :] = gFNC[i, :] / torch.std(gFNC[i, :])

    NegeEva = torch.zeros(K, dtype=torch_float)
    for i in range(K):
        NegeEva[i] = GIG_ICA_neg_entropy(gFNN[i, :])

    YR = (1.0 / dim_space) * torch.mm(Y, gFNN.t())

    ICOutMax = torch.zeros((K, dim_space), dtype=torch_float)

    for ICnum in range(K):
        reference = gFNN[ICnum, :]
        reference = reference.reshape([1, reference.shape[0]])  # keep first dimension

        # initialize wc
        wc = torch.mm(reference, Yinv).t()
        wc = wc / torch.norm(wc)  # [Nt, 1]

        # check spatial correspondence
        Source = torch.mm(wc.t(), Y)
        temp = mat_corr_torch(gFN, Source.t(), dataPrecision=dataPrecision)
        ps = torch.argmax(temp)
        if not ps == ICnum:
            print(f'\n Warning:\n  Initial pFN {ICnum} violates spatial correspondence constrain',
                  file=logFile, flush=True)

        y1 = torch.mm(wc.t(), Y)
        EyrInitial = (1 / dim_space) * torch.mm(y1, reference.t())
        NegeInitial = GIG_ICA_neg_entropy(y1)
        c = (torch.tan((EyrInitial * np.pi) / 2.0)) / NegeInitial
        IniObjValue = a * ErChuPai * torch.atan(c * NegeInitial) + b * EyrInitial

        for i in range(maxIter):
            Cosy1 = torch.cosh(y1)
            logCosy1 = torch.log(Cosy1)
            EGy1 = torch.mean(logCosy1)
            Negama = EGy1 - EGv
            tanhy1 = torch.tanh(y1)
            EYgy = (1 / dim_space) * torch.mm(Y, tanhy1.t())
            Jy1 = (EGy1 - EGv) ** 2
            KwDaoshu = ErChuPai * c * (1 / (1 + (c * Jy1) ** 2))
            Simgrad = YR[:, ICnum].reshape([YR.shape[0], 1])  # keep second dimension
            g = a * KwDaoshu * 2 * Negama * EYgy + b * Simgrad
            d = g / torch.norm(g)
            wx = wc + Nemda * d
            wx = wx / torch.norm(wx)
            y3 = torch.mm(wx.t(), Y)

            PreObjValue = a * ErChuPai * torch.atan(GIG_ICA_neg_entropy(y3)) + b * (1 / dim_space) * torch.mm(y3, reference.t())
            ObjValueChange = PreObjValue - IniObjValue

            dg = torch.mm(g.t(), d)
            ArmiCondiThr = Nemda * ftol * dg

            if torch.mm((wc - wx).t(), wc - wx) < error:
                break

            if ObjValueChange < 0:
                Nemda = Nemda / 2
                continue

            IniObjValue = PreObjValue
            y1 = y3
            wc = wx

            if ObjValueChange < ArmiCondiThr:
                Nemda = Nemda / 2
                continue

        Source = torch.mm(wx.t(), Y)
        ICOutMax[ICnum, :] = Source

    TCMax = (1 / dim_space) * torch.mm(Data, ICOutMax.t())

    FN = ICOutMax.t()
    TC = TCMax

    # back to numpy
    FN = FN.cpu().numpy()
    TC = TC.cpu().numpy()

    # check spatial correspondence
    temp = mat_corr(gFN, FN, dataPrecision)
    QC_Spatial_Correspondence = np.copy(np.diag(temp))
    temp -= np.diag(2 * np.ones(K))  # set diagonal values to lower than -1
    QC_Spatial_Correspondence_Control = np.max(temp, axis=1)
    QC_Delta_Sim = np.min(QC_Spatial_Correspondence - QC_Spatial_Correspondence_Control)
    if np.sum(QC_Delta_Sim < 0) > 0:
        print(f'\n Warning:\n  There are {np.sum(QC_Delta_Sim < 0)} pFNs violating spatial correspondence constrain',
              file=logFile, flush=True)

    print(f'\nFinished at ' +
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=logFile, flush=True)

    return TC, FN


def GIG_ICA_neg_entropy(x):
    y = torch.log(torch.cosh(x))
    E1 = torch.mean(y)
    E2 = torch.tensor(0.3745672075)
    negentropy = (E1 - E2) ** 2
    return negentropy


def pFN_GIG_ICA_cluster(dir_pnet_result: str, jobID=1):
    """
    Run the GIG-ICA for pFNs in cluster computation

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
    a = setting['FN_Computation']['Personalized_FN']['a']
    Nemda = setting['FN_Computation']['Personalized_FN']['Nemda']
    ftol = setting['FN_Computation']['Personalized_FN']['ftol']
    error = setting['FN_Computation']['Personalized_FN']['error']

    dataPrecision = setting['FN_Computation']['Computation']['dataPrecision']

    # separate maxIter and minIter for gFN and pFN
    if isinstance(maxIter, int) or (isinstance(maxIter, np.ndarray) and maxIter.shape == 1):
        maxIter_gFN = maxIter
        maxIter_pFN = maxIter
    else:
        maxIter_gFN = maxIter[0]
        maxIter_pFN = maxIter[1]

    # load data
    Data, CHeader, NHeader = load_fmri_scan(os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt'),
                          dataType=dataType, dataFormat=dataFormat,
                          Reshape=True, Brain_Mask=Brain_Mask, logFile=None)
    # perform NMF
    TC, pFN = pFN_GIG_ICA_torch(Data, gFN, maxIter=maxIter_pFN, a=a, Nemda=Nemda, ftol=ftol, error=error,
                                        dataPrecision=dataPrecision, logFile=None)

    if not isinstance(pFN, np.ndarray):
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
