# Yuncong Ma, 6/5/2024
# Quality control module of pNet using PyTorch

#########################################
# Packages

# to disable warnings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
# added by Yong Fan on July 23, 2024

# other functions of pNet
import numpy as np

from Module.Data_Input import *
from Basic.Matrix_Computation import *
from Module.Quality_Control import *
from Basic.Cluster_Computation import submit_bash_job


def run_quality_control_torch(dir_pnet_result: str):
    """
    Run the quality control module, which computes spatial correspondence and functional coherence
    The quality control result folder has consistent sub-folder organization as Personalized_FN
    Quality control results of each scan or combined scans are stored into sub-folders
    A single matlab file named Result.mat stores all quantitative values, including
    Spatial_Correspondence: spatial correspondence between pFNs and gFNs
    Delta_Spatial_Correspondence: the difference between spatial correspondence of matched pFNs-gFNs and miss-matched pFNs-gFNS
    Miss_Match: A 2D matrix, [N, 2], each row specifies which pFN is miss matched to a different gFN
    Functional_Coherence: weighted average of Pearson correlation between time series of pFNs and all nodes
    Functional_Coherence_Control: weighted average of Pearson correlation between time series of gFNs and all nodes
    A final report in txt format saved in the root directory of quality control folder
    It summaries the number of miss matched FNs for each failed scan

    :param dir_pnet_result: the directory of pNet result folder
    :return: None

    Yuncong Ma, 12/20/2023
    """

    # Setup sub-folders in pNet result
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, _ = setup_result_folder(dir_pnet_result)

    # Log file
    file_Final_Report = open(os.path.join(dir_pnet_QC, 'Final_Report.txt'), 'w')
    print('\nStart QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=file_Final_Report, flush=True)
    # Description of QC
    if file_Final_Report is not None:
        print_description_QC(file_Final_Report)

    setting = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    Data_Type = setting['Data_Type']
    Data_Format = setting['Data_Format']
    setting = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    combineScan = setting['Combine_Scan']
    dataPrecision = setting['Computation']['dataPrecision']

    # Information about scan list
    file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
    file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')

    list_scan = np.array([line.replace('\n', '') for line in open(file_scan, 'r')])
    list_subject_ID = np.array([line.replace('\n', '') for line in open(file_subject_ID, 'r')])
    subject_ID_unique = np.unique(list_subject_ID)
    N_Subject = subject_ID_unique.shape[0]
    list_subject_folder = np.array([line.replace('\n', '') for line in open(file_subject_folder, 'r')])
    list_subject_folder_unique = np.unique(list_subject_folder)

    # Load gFNs
    gFN = load_matlab_single_array(os.path.join(dir_pnet_gFN, 'FN.mat'))  # [dim_space, K]
    if Data_Type == 'Volume':
        Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
        gFN = reshape_FN(gFN, dataType=Data_Type, Brain_Mask=Brain_Mask)

    # data precision
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)

    # compute spatial correspondence and functional coherence for each scan
    if combineScan == 0:
        N_pFN = list_scan.shape[0]
    else:
        N_pFN = subject_ID_unique.shape[0]

    # Compute quality control measurement for each scan or scans combined
    flag_QC = 0
    for i in range(N_pFN):
        dir_pFN_indv = os.path.join(dir_pnet_pFN, list_subject_folder_unique[i])
        pFN = load_matlab_single_array(os.path.join(dir_pFN_indv, 'FN.mat'))
        if Data_Type == 'Volume':
            pFN = reshape_FN(pFN, dataType=Data_Type, Brain_Mask=Brain_Mask)

        # Get the scan list
        file_scan_list = os.path.join(dir_pFN_indv, 'Scan_List.txt')

        # Load the data
        if Data_Type == 'Surface':
            scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True, Normalization=None)

        elif Data_Type == 'Volume':
            scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                       Brain_Mask=Brain_Mask, Normalization=None)

        elif Data_Type == 'Surface-Volume':
            scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                       Normalization=None)

        else:
            raise ValueError('Unknown data type: ' + Data_Type)

        scan_data = torch.tensor(scan_data, dtype=torch_float)

        # Compute quality control measurement
        Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control =\
            compute_quality_control_torch(scan_data, gFN, pFN, dataPrecision=dataPrecision, logFile=None)

        # Finalize results
        Result = {'Spatial_Correspondence': Spatial_Correspondence,
                  'Delta_Spatial_Correspondence': Delta_Spatial_Correspondence,
                  'Miss_Match': Miss_Match,
                  'Functional_Coherence': Functional_Coherence,
                  'Functional_Coherence_Control': Functional_Coherence_Control}

        # Report the failed scans in the final report
        if Miss_Match.shape[0] > 0:
            flag_QC += 1
            print(' ' + str(Miss_Match.shape[0]) + ' miss matched FNs in sub folder: ' + list_subject_folder[i],
                  file=file_Final_Report, flush=True)

        # Save results
        dir_pFN_indv_QC = os.path.join(dir_pnet_QC, list_subject_folder[i])
        if not os.path.exists(dir_pFN_indv_QC):
            os.makedirs(dir_pFN_indv_QC)
        scipy.io.savemat(os.path.join(dir_pFN_indv_QC, 'Result.mat'), {'Result': Result}, do_compression=True)

    # Finish the final report
    if flag_QC == 0:
        print(f'\nSummary\n All {N_pFN} scans passed QC\n'
              f' This ensures that personalized FNs show highest spatial similarity to their group-level counterparts\n',
              file=file_Final_Report, flush=True)
    else:
        print(f'\nSummary\n Number of failed scans = {flag_QC}\n'
              f' This means those scans have at least one pFN show higher spatial similarity to a different group-level FN\n',
              file=file_Final_Report, flush=True)

    # Generate visualization
    visualize_quality_control(dir_pnet_result)

    print('\nFinished QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=file_Final_Report, flush=True)

    file_Final_Report.close()


def compute_quality_control_torch(scan_data, gFN, pFN, dataPrecision='double', logFile=None):
    """
    Compute quality control measurements, including spatial correspondence and functional coherence
    The spatial correspondence ensures one-to-one match between gFNs and pFNs
    The functional coherence ensures that pFNs gives better data fitting

    :param scan_data: 2D matrix, [dim_time, dim_space], np.ndarray or torch.Tensor
    :param gFN: 2D matrix, [dim_space, K], K is the number of FNs, np.ndarray or torch.Tensor
    :param pFN: 2D matrix, [dim_space, K], K is the number of FNs, np.ndarray or torch.Tensor
    :param dataPrecision: 'double' or 'single'
    :param logFile: None
    :return: Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control
    Outputs are numpy.ndarray
    Spatial correspondence is a 2D symmetric matrix [K, K], which measures the spatial correlation between gFNs and pFNs
    Delta_Spatial_Correspondence is a vector [K, ], which measures minimum difference of spatial correlation between matched and unmatched gFNs and pFNs
    Miss_Match is a 2D matrix [N, 2]. Each row notes a pair of miss-matched gFN and pFN.
    Functional_Coherence is a vector [K, ], which measures the weighted average correlation between node-wise fMRI signal in scan_data and time series of pFNs
    Functional_Coherence_Control is a vector [K, ], which measures the weighted average correlation between node-wise fMRI signal in scan_data and time series of gFNs

    Yuncong Ma, 2/9/2024
    """

    # data precision
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)
    if not isinstance(scan_data, torch.Tensor):
        scan_data = torch.tensor(scan_data, dtype=torch_float)
    else:
        scan_data = scan_data.type(torch_float)
    if not isinstance(gFN, torch.Tensor):
        gFN = torch.tensor(gFN, dtype=torch_float)
    else:
        gFN = gFN.type(torch_float)
    if not isinstance(pFN, torch.Tensor):
        pFN = torch.tensor(pFN, dtype=torch_float)
    else:
        pFN = pFN.type(torch_float)

    # Spatial correspondence
    K = gFN.shape[1]

    temp = mat_corr_torch(gFN, pFN, dataPrecision=dataPrecision)
    s_c = temp.cpu().numpy().copy()
    QC_Spatial_Correspondence = torch.clone(torch.diag(temp))
    temp -= torch.diag(2 * torch.ones(K))  # set diagonal values to lower than -1
    QC_Spatial_Correspondence_Control = torch.max(temp, dim=0)[0]
    QC_Delta_Sim = QC_Spatial_Correspondence - QC_Spatial_Correspondence_Control
   
    # Convert back to Numpy array
    Spatial_Correspondence = QC_Spatial_Correspondence.cpu().numpy()
    Delta_Spatial_Correspondence = QC_Delta_Sim.cpu().numpy()

    # Miss match between gFNs and pFNs
    # Index starts from 1
    if np.min(Delta_Spatial_Correspondence) >= 0:
        Miss_Match = np.empty((0,))
    else:
        ps = np.where(Delta_Spatial_Correspondence < 0)[0]
        #ps2 = np.asarray(np.argmax(Spatial_Correspondence, axis=0)).reshape(1, -1)[:, 0]
        ps2 = np.asarray(np.argmax(s_c, axis=0))
        Miss_Match = np.concatenate((ps[:, np.newaxis] + 1, ps2[ps[:], np.newaxis] + 1), axis=1)

    # Functional coherence
    # in case of negative spatial weightings, use np.abs
    pFN_signal = scan_data @ pFN / torch.sum(np.abs(pFN), dim=0, keepdims=True)
    
    Corr_FH = mat_corr_torch(pFN_signal, scan_data, dataPrecision=dataPrecision)
    Corr_FH[torch.isnan(Corr_FH)] = 0  # in case of zero signals
    Functional_Coherence = torch.sum(Corr_FH.T * pFN, dim=0) / torch.sum(np.abs(pFN), dim=0)
    # Use gFN as control
    gFN_signal = scan_data @ gFN / torch.sum(np.abs(pFN), dim=0, keepdims=True)
    
    Corr_FH = mat_corr_torch(gFN_signal, scan_data, dataPrecision=dataPrecision)
    Corr_FH[torch.isnan(Corr_FH)] = 0  # in case of zero signals
    Functional_Coherence_Control = torch.sum(Corr_FH.T * gFN, dim=0) / torch.sum(np.abs(gFN), dim=0)

    # Convert back to Numpy array
    Functional_Coherence = Functional_Coherence.numpy()
    Functional_Coherence_Control = Functional_Coherence_Control.numpy()

    return Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control


def run_quality_control_torch_cluster(dir_pnet_result: str):
    """
    Run the quality control module in cluster computation

    :param dir_pnet_result: the directory of pNet result folder
    :return: None

    Yuncong Ma, 6/5/2024
    """

    # Setup sub-folders in pNet result
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, _ = setup_result_folder(dir_pnet_result)

    # load setting
    settingCluster = load_json_setting(os.path.join(dir_pnet_dataInput, 'Cluster_Setting.json'))
    setting = {'Cluster': settingCluster}

    # Log file
    print('\nStart QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n', flush=True)

    settingCluster = load_json_setting(os.path.join(dir_pnet_dataInput, 'Cluster_Setting.json'))
    setting = {'Cluster': settingCluster}

    # Information about scan list
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
    list_subject_folder = np.array([line.replace('\n', '') for line in open(file_subject_folder, 'r')])
    list_subject_folder_unique = np.unique(list_subject_folder)

    nFolder = list_subject_folder_unique.shape[0]

    # submit jobs
    memory = setting['Cluster']['computation_resource']['memory_qc']
    n_thread = setting['Cluster']['computation_resource']['thread_qc']
    for rep in range(1, 1+nFolder):
        time.sleep(0.1)
        dir_indv = os.path.join(dir_pnet_QC, list_subject_folder_unique[rep-1])
        os.makedirs(dir_indv, exist_ok=True)
        if os.path.isfile(os.path.join(dir_indv, 'Result.mat')):
            continue
        submit_bash_job(dir_pnet_result,
                        python_command=f'pnet.compute_quality_control_torch_cluster(dir_pnet_result,{rep})',
                        memory=memory,
                        n_thread=n_thread,
                        bashFile=os.path.join(dir_indv, 'Cluster_job_bootstrap.sh'),
                        pythonFile=os.path.join(dir_indv, 'Cluster_job_bootstrap.py'),
                        logFile=os.path.join(dir_indv, 'Cluster_job_bootstrap.log')
                        )

    # check completion
    wait_time = 30
    flag_complete = np.zeros(nFolder)
    report_interval = 20
    Count = 0
    while np.sum(flag_complete) < nFolder:
        time.sleep(wait_time)
        Count += 1
        if Count % report_interval == 0:
            print(f'--> Found {np.sum(flag_complete)} finished jobs out of {nFolder} at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
        for rep in range(1, 1+nFolder):
            if flag_complete[rep-1] == 0 and os.path.isfile(os.path.join(dir_pnet_QC, list_subject_folder_unique[rep-1], 'Result.mat')):
                flag_complete[rep-1] = 1

    # gather results and generate final report
    file_Final_Report = open(os.path.join(dir_pnet_QC, 'Final_Report.txt'), 'w')
    print('\nStart QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=file_Final_Report, flush=True)
    # Description of QC
    if file_Final_Report is not None:
        print_description_QC(file_Final_Report)
    flag_QC = 0
    for i in range(1, 1+nFolder):
        Result = load_matlab_single_variable(os.path.join(dir_pnet_QC, list_subject_folder_unique[i-1], 'Result.mat'))
        if Result['Miss_Match'][0, 0].shape[0] > 0:
            flag_QC += 1
            print(' ' + str(Result['Miss_Match'][0, 0].shape[0]) + ' miss matched FNs in sub folder: ' + list_subject_folder_unique[i-1],
                  file=file_Final_Report, flush=True)
    # Finish the final report
    if flag_QC == 0:
        print(f'\nSummary\n All {nFolder} scans passed QC\n'
              f' This ensures that personalized FNs show highest spatial similarity to their group-level counterparts\n',
              file=file_Final_Report, flush=True)
    else:
        print(f'\nSummary\n Number of failed scans = {flag_QC}\n'
              f' This means those scans have at least one pFN show higher spatial similarity to a different group-level FN\n',
              file=file_Final_Report, flush=True)

    file_Final_Report.close()

    # Generate visualization
    visualize_quality_control(dir_pnet_result)

    print('\nFinished QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n', flush=True)


def compute_quality_control_torch_cluster(dir_pnet_result: str, jobID=1):
    """
    Run the QC in cluster computation

    :param dir_pnet_result: directory of pNet result folder
    :param jobID: jobID starting from 1
    :return: None

    Yuncong Ma, 12/1/2023
    """

    # Setup sub-folders in pNet result
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, _ = setup_result_folder(dir_pnet_result)

    print('\nStart QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n', flush=True)

    setting = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    Data_Type = setting['Data_Type']
    Data_Format = setting['Data_Format']
    setting = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    dataPrecision = setting['Computation']['dataPrecision']

    # Information about scan list
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')
    list_subject_folder = np.array([line.replace('\n', '') for line in open(file_subject_folder, 'r')])
    list_subject_folder_unique = np.unique(list_subject_folder)
    dir_pFN_indv = os.path.join(dir_pnet_pFN, list_subject_folder_unique[jobID-1])

    # Load gFNs
    gFN = load_matlab_single_array(os.path.join(dir_pnet_gFN, 'FN.mat'))  # [dim_space, K]
    if Data_Type == 'Volume':
        Brain_Mask = load_brain_template(os.path.join(dir_pnet_dataInput, 'Brain_Template.json.zip'))['Brain_Mask']
        gFN = reshape_FN(gFN, dataType=Data_Type, Brain_Mask=Brain_Mask)

    # data precision
    torch_float, torch_eps = set_data_precision_torch(dataPrecision)

    pFN = load_matlab_single_array(os.path.join(dir_pFN_indv, 'FN.mat'))
    if Data_Type == 'Volume':
        pFN = reshape_FN(pFN, dataType=Data_Type, Brain_Mask=Brain_Mask)

    # Get the scan list
    file_scan_list = os.path.join(dir_pFN_indv, 'Scan_List.txt')

    # Load the data
    if Data_Type == 'Surface':
        scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True, Normalization=None)

    elif Data_Type == 'Volume':
        scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                   Brain_Mask=Brain_Mask, Normalization=None)

    elif Data_Type == 'Surface-Volume':
        scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                   Normalization=None)
    else:
        raise ValueError('Unknown data type: ' + Data_Type)

    scan_data = torch.tensor(scan_data, dtype=torch_float)

    # Compute quality control measurement
    Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control =\
        compute_quality_control_torch(scan_data, gFN, pFN, dataPrecision=dataPrecision, logFile=None)

    # Finalize results
    Result = {'Spatial_Correspondence': Spatial_Correspondence,
              'Delta_Spatial_Correspondence': Delta_Spatial_Correspondence,
              'Miss_Match': Miss_Match,
              'Functional_Coherence': Functional_Coherence,
              'Functional_Coherence_Control': Functional_Coherence_Control}

    # Save results
    dir_pFN_indv_QC = os.path.join(dir_pnet_QC, list_subject_folder_unique[jobID-1])
    os.makedirs(dir_pFN_indv_QC, exist_ok=True)
    scipy.io.savemat(os.path.join(dir_pFN_indv_QC, 'Result.mat'), {'Result': Result}, do_compression=True)