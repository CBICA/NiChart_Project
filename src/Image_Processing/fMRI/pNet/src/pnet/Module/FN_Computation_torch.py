# Yuncong Ma, 6/5/2024
# FN Computation module of pNet
# Pytorch version

#########################################
# Packages
import scipy.io as sio

# other functions of pNet
from Module.Data_Input import *
from Module.FN_Computation import check_gFN, bootstrap_scan, setup_pFN_folder
import Module.GIG_ICA as GIG_ICA
import Module.SR_NMF as SR_NMF
from Basic.Cluster_Computation import submit_bash_job


class setup_FN_Computation:

    setup_SR_NMF = SR_NMF.setup_SR_NMF

    setup_GIG_ICA = GIG_ICA.setup_GIG_ICA


def run_FN_Computation_torch(dir_pnet_result: str):
    """
    run the FN Computation module with settings ready in Data_Input and FN_Computation

    :param dir_pnet_result: directory of pNet result folder

    Yuncong Ma, 2/8/2024
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # log file
    logFile_FNC = os.path.join(dir_pnet_FNC, 'log.log')
    logFile_FNC = open(logFile_FNC, 'w')
    print('\nStart FN computation using PyTorch at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
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

    # FN method
    FN_Method = setting['FN_Computation']['Method']

    if FN_Method == 'SR-NMF':
        # ============== SR-NMF ============== #
        print('FN computation uses sparsity-regularized non-negative matrix factorization method', file=logFile_FNC, flush=True)
        # Generate additional parameters
        gNb = SR_NMF.compute_gNb(Brain_Template)
        scipy.io.savemat(os.path.join(dir_pnet_FNC, 'gNb.mat'), {'gNb': gNb}, do_compression=True)

        # ============== gFN Computation ============== #
        if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
            # 2 steps
            # step 1 ============== bootstrap
            # sub-folder in FNC for storing bootstrapped results
            print('Start to prepare bootstrap files', file=logFile_FNC, flush=True)
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            if not os.path.exists(dir_pnet_BS):
                os.makedirs(dir_pnet_BS)
            # Log
            logFile = os.path.join(dir_pnet_BS, 'Log_gFN_SR_NMF.log')

            # Input files
            file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
            file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
            file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
            file_group_ID = os.path.join(dir_pnet_dataInput, 'Group_ID.txt')
            if not os.path.exists(file_group_ID):
                file_group = None
            # Parameters
            combineScan = setting['FN_Computation']['Combine_Scan']
            init = setting['FN_Computation']['Group_FN']['BootStrap']['init']  #added on 08/03/2024
            samplingMethod = setting['FN_Computation']['Group_FN']['BootStrap']['samplingMethod']
            sampleSize = setting['FN_Computation']['Group_FN']['BootStrap']['sampleSize']
            nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']
            nTPoints = setting['FN_Computation']['Group_FN']['BootStrap']['nTPoints']   #added on 08/01/2024

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

            # NMF on bootstrapped subsets
            print('Start to NMF for each bootstrap at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            K = setting['FN_Computation']['K']
            nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']
            for rep in range(1, 1+nBS):
                # log file
                logFile = os.path.join(dir_pnet_BS, str(rep), 'Log_gFN_SR_NMF.log')
                # load data
                file_scan_list = os.path.join(dir_pnet_BS, str(rep), 'Scan_List.txt')
                Data, CHeader, NHeader = load_fmri_scan(file_scan_list, dataType=dataType, dataFormat=dataFormat, nTPoints=nTPoints, Reshape=True, Brain_Mask=Brain_Mask,
                                      Normalization='vp-vmax', logFile=logFile)
                # perform NMF
                FN_BS = SR_NMF.gFN_SR_NMF_torch(Data, K, gNb, init=init, maxIter=maxIter_gFN, minIter=minIter_gFN, error=error, normW=normW,
                                                Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL, vxI=vxI, ard=ard, eta=eta,
                                                nRepeat=nRepeat, dataPrecision=dataPrecision, logFile=logFile)
                # save results
                FN_BS = reshape_FN(FN_BS.numpy(), dataType=dataType, Brain_Mask=Brain_Mask)
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
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            K = setting['FN_Computation']['K']
            nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']
            FN_BS = np.empty(nBS, dtype=np.ndarray)
            # load bootstrapped results
            for rep in range(1, nBS+1):
                FN_BS[rep-1] = np.array(reshape_FN(load_matlab_single_array(os.path.join(dir_pnet_BS, str(rep), 'FN.mat')), dataType=dataType, Brain_Mask=Brain_Mask))
            gFN_BS = np.concatenate(FN_BS, axis=1)
            # log
            logFile = os.path.join(dir_pnet_gFN, 'Log.log')
            # Fuse bootstrapped results
            gFN = SR_NMF.gFN_fusion_NCut_torch(gFN_BS, K, logFile=logFile)
            # output
            gFN = reshape_FN(gFN.numpy(), dataType=dataType, Brain_Mask=Brain_Mask)
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
            check_gFN(gFN, method=FN_Method, logFile=logFile_FNC)
            if dataType == 'Volume':
                gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
            # save FNs in nii.gz and TC as txt file  FY 07/26/2024
            # won't save imaging data for pre-computed Group FNs
            #output_FN(FN=gFN,
            #          file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
            #          file_brain_template = Brain_Template,
            #          dataFormat=dataFormat)

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

            # log file
            logFile = os.path.join(dir_pnet_pFN_indv, 'Log_pFN_SR_NMF.log')
            # load data
            Data, CHeader, NHeader = load_fmri_scan(os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt'),
                                  dataType=dataType, dataFormat=dataFormat,
                                  Reshape=True, Brain_Mask=Brain_Mask, logFile=logFile)
            # perform NMF
            TC, pFN = SR_NMF.pFN_SR_NMF_torch(Data, gFN, gNb, maxIter=maxIter_pFN, minIter=minIter_pFN, meanFitRatio=meanFitRatio,
                                              error=error, normW=normW,
                                              Alpha=Alpha, Beta=Beta, alphaS=alphaS, alphaL=alphaL,
                                              vxI=vxI, ard=ard, eta=eta,
                                              dataPrecision=dataPrecision, logFile=logFile)
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


    elif FN_Method == 'GIG-ICA':
        # ============== GIG-ICA ============== #
        # gFNs must be loaded
        if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
            print(f'\nError: \nCannot find the group-level FNs for GIG-ICA ', file=logFile_FNC, flush=True)
            raise ValueError('Require gFNs for GIG-ICA')
        file_gFN = setting['FN_Computation']['Group_FN']['file_gFN']
        #gFN = load_matlab_single_array(file_gFN) # have to be 'Volume'
        gFN, _, _ = load_fmri_scan(file_gFN, dataType='Volume', dataFormat='Volume (*.nii, *.nii.gz, *.mat)', Reshape=False, Normalization=None)
        if dataType == 'Volume':
            Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
            gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
        check_gFN(gFN, method=FN_Method, logFile=logFile_FNC)
        if dataType == 'Volume':
            gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
        sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
        # save FNs in nii.gz and TC as txt file  FY 07/26/2024
        # won't save precomputed gFN
        #output_FN(FN=gFN,
        #          file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
        #          file_brain_template = Brain_Template,
        #          dataFormat=dataFormat)
        print('load precomputed gFNs', file=logFile_FNC, flush=True)

        # ============== pFN Computation ============== #
        print('Start to compute pFNs at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)
        # load precomputed gFNs
        gFN = load_matlab_single_array(os.path.join(dir_pnet_gFN, 'FN.mat'))
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

            # log file
            logFile = os.path.join(dir_pnet_pFN_indv, 'Log_pFN_GIG_ICA.log')
            # load data
            Data, CHeader, NHeader = load_fmri_scan(os.path.join(dir_pnet_pFN_indv, 'Scan_List.txt'),
                                  dataType=dataType, dataFormat=dataFormat,
                                  Reshape=True, Brain_Mask=Brain_Mask, logFile=logFile)

            # perform GIG-ICA
            TC, pFN = GIG_ICA.pFN_GIG_ICA_torch(Data, gFN, maxIter=maxIter_pFN, a=a, Nemda=Nemda, ftol=ftol, error=error,
                                                dataPrecision=dataPrecision, logFile=logFile)
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
        # ============================================= #
        print('Finished FN computation at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=logFile_FNC, flush=True)

def run_FN_computation_torch_cluster(dir_pnet_result: str):
    """
    run the FN Computation module in cluster

    :param dir_pnet_result: directory of pNet result folder

    Yuncong Ma, 6/5/2024
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, _, _ = setup_result_folder(dir_pnet_result)

    # log file
    print('\nStart FN computation using PyTorch at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n', flush=True)

    # load settings for data input and FN computation
    if not os.path.isfile(os.path.join(dir_pnet_dataInput, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder Data_Input')
    if not os.path.isfile(os.path.join(dir_pnet_FNC, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder FN_Computation')
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingCluster = load_json_setting(os.path.join(dir_pnet_dataInput, 'Cluster_Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC, 'Cluster': settingCluster}
    print('Settings are loaded from folder Data_Input, FN_Computation and Cluster', flush=True)

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

    # FN method
    FN_Method = setting['FN_Computation']['Method']

    if FN_Method == 'SR-NMF':
        # ============== SR-NMF ============== #
        print('FN computation uses spatial-regularized non-negative matrix factorization method', flush=True)

        # Generate additional parameters
        if not os.path.isfile(os.path.join(dir_pnet_FNC, 'gNb.mat')):
            gNb = SR_NMF.compute_gNb(Brain_Template)
            scipy.io.savemat(os.path.join(dir_pnet_FNC, 'gNb.mat'), {'gNb': gNb}, do_compression=True)

        # ============== gFN Computation ============== #
        if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
            # 2 steps
            # step 1 ============== bootstrap
            # sub-folder in FNC for storing bootstrapped results
            print('Start to prepare bootstrap files', flush=True)
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            if not os.path.exists(dir_pnet_BS):
                os.makedirs(dir_pnet_BS)

            # Input files
            file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
            file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
            file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
            file_group_ID = os.path.join(dir_pnet_dataInput, 'Group_ID.txt')
            if not os.path.exists(file_group_ID):
                file_group_ID = None
            # Parameters
            combineScan = setting['FN_Computation']['Combine_Scan']
            samplingMethod = setting['FN_Computation']['Group_FN']['BootStrap']['samplingMethod']
            sampleSize = setting['FN_Computation']['Group_FN']['BootStrap']['sampleSize']
            nBS = setting['FN_Computation']['Group_FN']['BootStrap']['nBS']

            # create scan lists for bootstrap
            flag_complete = np.zeros(nBS)
            for i in range(1, 1+nBS):
                if os.path.exists(os.path.join(dir_pnet_BS, str(i))) and os.path.isfile(os.path.join(dir_pnet_BS, str(i), 'Scan_List.txt')):
                    flag_complete[i-1] = 1
            if np.sum(flag_complete) < nBS:
                bootstrap_scan(dir_pnet_BS, file_scan, file_subject_ID, file_subject_folder,
                                 file_group_ID=file_group_ID, combineScan=combineScan,
                                 samplingMethod=samplingMethod, sampleSize=sampleSize, nBS=nBS, logFile=None)

            # submit jobs
            memory = setting['Cluster']['computation_resource']['memory_bootstrap']
            n_thread = setting['Cluster']['computation_resource']['thread_bootstrap']
            for rep in range(1, 1+nBS):
                time.sleep(0.1)
                if os.path.isfile(os.path.join(dir_pnet_BS, str(i), 'FN.mat')):
                    continue
                submit_bash_job(dir_pnet_result,
                                python_command=f'pnet.SR_NMF.gFN_SR_NMF_boostrapping_cluster(dir_pnet_result,{rep})',
                                memory=memory,
                                n_thread=n_thread,
                                bashFile=os.path.join(dir_pnet_BS, str(rep), 'cluster_job_bootstrap.sh'),
                                pythonFile=os.path.join(dir_pnet_BS, str(rep), 'cluster_job_bootstrap.py'),
                                logFile=os.path.join(dir_pnet_BS, str(rep), 'cluster_job_bootstrap.log')
                                )

            # check completion
            wait_time = 300
            flag_complete = np.zeros(nBS)
            dir_pnet_BS = os.path.join(dir_pnet_FNC, 'BootStrapping')
            report_interval = 12
            Count = 0
            while np.sum(flag_complete) < nBS:
                time.sleep(wait_time)
                Count += 1
                if Count % report_interval == 0:
                    print(f'--> Found {np.sum(flag_complete)} finished jobs out of {nBS} at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
                for rep in range(1, 1+nBS):
                    if flag_complete[rep-1] == 0 and os.path.isfile(os.path.join(dir_pnet_BS, str(rep), 'FN.mat')):
                        flag_complete[rep-1] = 1

            # step 2 ============== fuse results
            # Generate gFNs
            print('Start to fuse bootstrapped results using NCut at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
            memory = setting['Cluster']['computation_resource']['memory_fusion']
            n_thread = setting['Cluster']['computation_resource']['thread_fusion']

            if not os.path.isfile(os.path.join(dir_pnet_gFN, 'FN.mat')):
                submit_bash_job(dir_pnet_result,
                                python_command='pnet.SR_NMF.fuse_FN_cluster(dir_pnet_result)',
                                memory=memory,
                                n_thread=n_thread,
                                bashFile=os.path.join(dir_pnet_BS, 'cluster_job_fusion.sh'),
                                pythonFile=os.path.join(dir_pnet_BS, 'cluster_job_fusion.py'),
                                logFile=os.path.join(dir_pnet_BS, 'cluster_job_fusion.log')
                                )

            # check completion
            wait_time = 60
            report_interval = 30
            flag_complete = 0
            Count = 0
            while flag_complete == 0:
                time.sleep(wait_time)
                Count += 1
                if Count % report_interval == 0:
                    print(f'--> Found {np.sum(flag_complete)} finished jobs out of 1 at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
                if os.path.isfile(os.path.join(dir_pnet_gFN, 'FN.mat')):
                    flag_complete = 1
                    break

        else:  # use precomputed gFNs
            file_gFN = setting['FN_Computation']['Group_FN']['file_gFN']
            gFN, _, _ = load_matlab_single_array(file_gFN)
            if dataType == 'Volume':
                Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
                gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            check_gFN(gFN, method=FN_Method)
            if dataType == 'Volume':
                gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
            sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
            # save FNs in nii.gz and TC as txt file  FY 07/26/2024
            # won't save precomputed gFN
            #output_FN(FN=gFN,
            #          file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
            #          file_brain_template = Brain_Template,
            #          dataFormat=dataFormat)
            print('load precomputed gFNs', flush=True)
        # ============================================= #

        # ============== pFN Computation ============== #
        print('Start to compute pFNs at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
        # setup folders in Personalized_FN
        list_subject_folder = setup_pFN_folder(dir_pnet_result)
        nScan = len(list_subject_folder)

        # submit jobs
        memory = setting['Cluster']['computation_resource']['memory_pFN']
        n_thread = setting['Cluster']['computation_resource']['thread_pFN']
        for scan in range(1, 1+nScan):
            time.sleep(0.1)
            if os.path.isfile(os.path.join(dir_pnet_pFN, list_subject_folder[scan-1], 'FN.mat')):
                continue
            submit_bash_job(dir_pnet_result,
                            python_command=f'pnet.SR_NMF.pFN_SR_NMF_cluster(dir_pnet_result,{scan})',
                            memory=memory,
                            n_thread=n_thread,
                            bashFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan-1], 'cluster_job_pFN.sh'),
                            pythonFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan-1], 'cluster_job_pFN.py'),
                            logFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan-1], 'cluster_job_pFN.log')
                            )
        # check completion
        wait_time = 30
        report_interval = 120
        flag_complete = np.zeros(nScan)
        Count = 0
        while np.sum(flag_complete) < nScan:
            time.sleep(wait_time)
            Count += 1
            if Count % report_interval == 0:
                print(f'--> Found {np.sum(flag_complete)} finished jobs out of {nScan} at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
            for scan in range(1, 1+nScan):
                if flag_complete[scan-1] == 0 and os.path.isfile(os.path.join(dir_pnet_pFN, list_subject_folder[scan-1], 'FN.mat')):
                    flag_complete[scan-1] = 1
        # ============================================= #

    elif FN_Method == 'GIG-ICA':
        # ============== GIG-ICA ============== #
        # gFNs must be loaded
        if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
            print(f'\nError: \nCannot find the group-level FNs for GIG-ICA ', flush=True)
            raise ValueError('Require gFNs for GIG-ICA')
        file_gFN = setting['FN_Computation']['Group_FN']['file_gFN']
        #gFN = load_matlab_single_array(file_gFN) # have to be 'Volume'
        gFN = load_fmri_scan(file_gFN, dataType='Volume', dataFormat='Volume (*.nii, *.nii.gz, *.mat)', Reshape=False, Normalization=None)
        if dataType == 'Volume':
            Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
            gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
        check_gFN(gFN, method=FN_Method)
        if dataType == 'Volume':
            gFN = reshape_FN(gFN, dataType=dataType, Brain_Mask=Brain_Mask)
        sio.savemat(os.path.join(dir_pnet_gFN, 'FN.mat'), {"FN": gFN}, do_compression=True)
        # save FNs in nii.gz and TC as txt file  FY 07/26/2024
        # won't save pre-computed gFNs
        #output_FN(FN=gFN,
        #          file_output=os.path.join(dir_pnet_gFN, 'FN.mat'),
        #          file_brain_template = Brain_Template,
        #          dataFormat=dataFormat, Cheader = CHeader, Nheader = NHeader)

        print('load precomputed gFNs', flush=True)

        # ============== pFN Computation ============== #
        print('Start to compute pFNs at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              flush=True)
        # setup folders in Personalized_FN
        list_subject_folder = setup_pFN_folder(dir_pnet_result)
        nScan = len(list_subject_folder)

        # submit jobs
        memory = setting['Cluster']['computation_resource']['memory_pFN']
        n_thread = setting['Cluster']['computation_resource']['thread_pFN']
        for scan in range(1, 1 + nScan):
            time.sleep(0.1)
            if os.path.isfile(os.path.join(dir_pnet_pFN, list_subject_folder[scan - 1], 'FN.mat')):
                continue
            submit_bash_job(dir_pnet_result,
                            python_command=f'pnet.GIG_ICA.pFN_GIG_ICA_cluster(dir_pnet_result,{scan})',
                            memory=memory,
                            n_thread=n_thread,
                            bashFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan - 1], 'cluster_job_pFN.sh'),
                            pythonFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan - 1],
                                                    'cluster_job_pFN.py'),
                            logFile=os.path.join(dir_pnet_pFN, list_subject_folder[scan - 1], 'cluster_job_pFN.log')
                            )

        # check completion
        wait_time = 30
        report_interval = 120
        flag_complete = np.zeros(nScan)
        Count = 0
        while np.sum(flag_complete) < nScan:
            time.sleep(wait_time)
            Count += 1
            if Count % report_interval == 0:
                print(f'--> Found {np.sum(flag_complete)} finished jobs out of {nScan} at ' + time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
            for scan in range(1, 1 + nScan):
                if flag_complete[scan - 1] == 0 and os.path.isfile(
                        os.path.join(dir_pnet_pFN, list_subject_folder[scan - 1], 'FN.mat')):
                    flag_complete[scan - 1] = 1
        # ============================================= #

    print('Finished FN computation at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
