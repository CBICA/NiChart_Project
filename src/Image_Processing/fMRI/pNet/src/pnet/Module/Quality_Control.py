# Yuncong Ma, 2/14/2024
# Quality control module of pNet

#########################################
# Packages
import pandas as pd
import pandas.api.types as pdtypes
from plotnine import (
    ggplot,
    aes,
    stage,
    geom_violin,
    geom_point,
    geom_line,
    geom_boxplot,
    scale_fill_manual,
    theme,
    theme_classic,
    element_text,
    element_line
)

# to disable warningings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
# added by Yong Fan on July 23, 2024

# other functions of pNet
from Module.Data_Input import *
from Basic.Matrix_Computation import *


def print_description_QC(logFile: str):
    """
    Print the description of quality control module

    :param logFile:

    Yuncong Ma, 9/28/2023
    """

    print('\nQuality control module checks the spatial correspondence and functional coherence.\n'
          'The spatial correspondence measures the spatial similarity between pFNs and gFNs.\n'
          'pFNs are supposed to have the highest spatial similarity to their group-level counterparts, otherwise violating the QC.\n'
          'The functional coherence measures the average temporal correlation between time series of each pFN and the whole brain.\n'
          'pFNs are supposed to show improved functional coherence compared to gFNs.\n', file=logFile, flush=True)


def run_quality_control(dir_pnet_result: str):
    """
    run_quality_control(dir_pnet_result: str)
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

    Yuncong Ma, 12/19/2023
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
    np_float, np_eps = set_data_precision(dataPrecision)

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
            #scan_data = scan_data.astype(np_float)

        elif Data_Type == 'Volume':
            scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                       Brain_Mask=Brain_Mask, Normalization=None)
            #scan_data = scan_data.astype(np_float)


        elif Data_Type == 'Surface-Volume':
            scan_data, _, _ = load_fmri_scan(file_scan_list, dataType=Data_Type, dataFormat=Data_Format, Reshape=True,
                                       Normalization=None)

        else:
            raise ValueError('Unknown data type: ' + Data_Type)

        # Compute quality control measurement
        Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control =\
            compute_quality_control(scan_data, gFN, pFN, dataPrecision=dataPrecision, logFile=None)

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

    #file_Final_Report.close()

    # Generate visualization
    visualize_quality_control(dir_pnet_result)

    print('\nFinished QC at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
          file=file_Final_Report, flush=True)
    file_Final_Report.close()


def compute_quality_control(scan_data: np.ndarray, gFN: np.ndarray, pFN: np.ndarray, dataPrecision='double', logFile=None):
    """
    compute_quality_control(scan_data: np.ndarray, gFN: np.ndarray, pFN: np.ndarray, dataPrecision='double', logFile=None)
    Compute quality control measurements, including spatial correspondence and functional coherence
    The spatial correspondence ensures one-to-one match between gFNs and pFNs
    The functional coherence ensures that pFNs gives better data fitting

    :param scan_data: 2D matrix, [dim_time, dim_space]
    :param gFN: 2D matrix, [dim_space, K], K is the number of FNs
    :param pFN: 2D matrix, [dim_space, K], K is the number of FNs
    :param dataPrecision: 'double' or 'single'
    :param logFile: None
    :return: Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control
    Spatial correspondence is a 2D symmetric matrix [K, K], which measures the spatial correlation between gFNs and pFNs
    Delta_Spatial_Correspondence is a vector [K, ], which measures minimum difference of spatial correlation between matched and unmatched gFNs and pFNs
    Miss_Match is a 2D matrix [N, 2]. Each row notes a pair of miss-matched gFN and pFN.
    Functional_Coherence is a vector [K, ], which measures the weighted average correlation between node-wise fMRI signal in scan_data and time series of pFNs
    Functional_Coherence_Control is a vector [K, ], which measures the weighted average correlation between node-wise fMRI signal in scan_data and time series of gFNs

    Yuncong Ma, 12/6/2023
    """

    # Spatial correspondence
    K = gFN.shape[1]

    temp = mat_corr(gFN, pFN, dataPrecision=dataPrecision)
    s_c = temp.copy()
    QC_Spatial_Correspondence = np.diag(temp).copy()

    temp -= np.diag(2 * np.ones(K))  # set diagonal values to lower than -1
    QC_Spatial_Correspondence_Control = np.max(temp, axis=0)

    QC_Delta_Sim = QC_Spatial_Correspondence - QC_Spatial_Correspondence_Control

    # set  back to Numpy array
    Spatial_Correspondence = QC_Spatial_Correspondence
    Delta_Spatial_Correspondence = QC_Delta_Sim

    #Spatial_Correspondence = mat_corr(gFN, pFN, dataPrecision=dataPrecision)
    #Delta_Spatial_Correspondence = np.diag(Spatial_Correspondence) - np.max(
    #    Spatial_Correspondence - np.diag(2 * np.ones(K)), axis=0)

    # Miss match between gFNs and pFNs
    # Index starts from 1
    if np.min(Delta_Spatial_Correspondence) >= 0:
        Miss_Match = np.empty((0,))
    else:
        ps = np.where(Delta_Spatial_Correspondence < 0)[0]
        #ps2 = np.argmax(Spatial_Correspondence, axis=0)
        ps2 = np.asarray(np.argmax(s_c, axis=0)) #.reshape(1, -1)[:, 0]
        Miss_Match = np.concatenate((ps[:, np.newaxis] + 1, ps2[ps[:], np.newaxis] + 1), axis=1)

        #np.concatenate((ps[:, np.newaxis] + 1, ps2[:, np.newaxis] + 1), axis=1)

    # Functional coherence
    pFN_signal = scan_data @ pFN / np.sum(np.abs(pFN), axis=0, keepdims=True)
    Corr_FH = mat_corr(pFN_signal, scan_data, dataPrecision=dataPrecision)
    Corr_FH[np.isnan(Corr_FH)] = 0  # in case of zero signals
    Functional_Coherence = np.sum(Corr_FH.T * pFN, axis=0) / np.sum(np.abs(pFN), axis=0)
    # Use gFN as control
    gFN_signal = scan_data @ gFN / np.sum(np.abs(pFN), axis=0, keepdims=True)
    Corr_FH = mat_corr(gFN_signal, scan_data, dataPrecision=dataPrecision)
    Corr_FH[np.isnan(Corr_FH)] = 0  # in case of zero signals
    Functional_Coherence_Control = np.sum(Corr_FH.T * gFN, axis=0) / np.sum(np.abs(gFN), axis=0)

    return Spatial_Correspondence, Delta_Spatial_Correspondence, Miss_Match, Functional_Coherence, Functional_Coherence_Control


def visualize_quality_control(dir_pnet_result: str):
    """
    Visualize the results of quality control for all scans in the dataset
    :param dir_pnet_result: directory of the pNet result folder
    :return: None

    Yuncong Ma, 2/14/2024
    """

    # Setup sub-folders in pNet result
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, _ = setup_result_folder(dir_pnet_result)

    # get settings
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    K = settingFNC['K']
    Combine_Scan = settingDataInput['Combine_Scan']

    # folder info
    list_subject_folder = load_txt_list(os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt'))
    list_subject_folder_unique = np.unique(list_subject_folder)
    nFolder = list_subject_folder_unique.shape[0]

    # load results
    Spatial_Correspondence = np.zeros((nFolder, K))
    Delta_Spatial_Correspondence = np.zeros((nFolder, K))
    Functional_Coherence = np.zeros((nFolder, K))
    Functional_Coherence_Control = np.zeros((nFolder, K))
    for i in range(nFolder):
        dir_indv = os.path.join(dir_pnet_QC, list_subject_folder_unique[i])
        Result = load_matlab_single_variable(os.path.join(dir_indv, 'Result.mat'))

        # get spatial correspondence and functional coherence using pFNs and gFNs
        # Add support for previous terminology 'Functional_Homogeneity'
        if Result['Spatial_Correspondence'][0, 0].shape[1] == 1:  # For some correlation matrix between gFN and pFN
            Spatial_Correspondence[i, :] = np.diag(Result['Spatial_Correspondence'][0, 0])
        else:
            Spatial_Correspondence[i, :] = Result['Spatial_Correspondence'][0, 0]
        Delta_Spatial_Correspondence[i, :] = Result['Delta_Spatial_Correspondence'][0, 0]
        if 'Functional_Coherence' in Result.dtype.names:
            Functional_Coherence[i, :] = Result['Functional_Coherence'][0, 0]
            Functional_Coherence_Control[i, :] = Result['Functional_Coherence_Control'][0, 0]
        else:
            Functional_Coherence[i, :] = Result['Functional_Homogeneity'][0, 0]
            Functional_Coherence_Control[i, :] = Result['Functional_Homogeneity_Control'][0, 0]

    # output results
    Result = {'Spatial_Correspondence': Spatial_Correspondence,
              'Delta_Spatial_Correspondence': Delta_Spatial_Correspondence,
              'Functional_Coherence': Functional_Coherence,
              'Functional_Coherence_Control': Functional_Coherence_Control}

    sio.savemat(os.path.join(dir_pnet_QC, 'Result.mat'), {'Result': Result}, do_compression=True)
    

    # visualization for spatial correspondence
    # minimum of delta spatial correspondence
    min_delta_SC = np.min(Delta_Spatial_Correspondence, axis=1)
    min_delta_SC[np.isnan(min_delta_SC)] = 0
    passed = min_delta_SC[min_delta_SC >= 0]
    ps_passed = np.where(min_delta_SC >= 0)[0]

    if Combine_Scan:
        Axes_Name = ['Subjects Passed QC', 'Delta Spatial Correspondence']
    else:
        Axes_Name = ['Scans Passed QC', 'Delta Spatial Correspondence']
    Group_Name = ['']
    Group_Color = ['dodgerblue']

    n = passed.shape[0]

    df = pd.DataFrame({
        Axes_Name[1]: np.hstack([passed]),
        Axes_Name[0]: np.repeat(Group_Name, n),
        'id': np.hstack([range(n)])
    })

    df[Axes_Name[0]] = df[Axes_Name[0]].astype(pdtypes.CategoricalDtype(categories=Group_Name))
    df.head()

    line_size = 0.6
    # set the transparency for filling area
    fill_alpha = 0.8
    point_alpha = 0.5

    df[Axes_Name[0]] = df[Axes_Name[0]].astype(pdtypes.CategoricalDtype(categories=Group_Name))
    df.head()

    shift = 0.1

    def alt_sign(x):
        return (-1) ** x

    m1 = aes(x=stage(Axes_Name[0], after_scale='x+shift*alt_sign(x)'))              # shift outward
    m2 = aes(x=stage(Axes_Name[0], after_scale='x-shift*alt_sign(x)'))              # shift inward

    df_additional = pd.DataFrame({Axes_Name[0]: [0, 1], Axes_Name[1]: [0, 0]})

    Figure = (ggplot(df, aes(Axes_Name[0], Axes_Name[1], fill=Axes_Name[0]))
     + geom_violin(m1, style='left-right', alpha=fill_alpha, size=line_size, show_legend=False)
     + geom_boxplot(width=shift, alpha=fill_alpha, size=line_size, outlier_alpha=point_alpha, show_legend=False)
     + geom_point(m2, color='none', alpha=point_alpha, size=2, show_legend=False)
     + scale_fill_manual(values=Group_Color)
     + theme_classic()
     + theme(figure_size=(4, 4),
             axis_title=element_text(family='Arial', size=16, weight='bold', color='black'),
             axis_text=element_text(family='Arial', size=14, weight='bold', color='black'),
             axis_line=element_line(size=2, color='black'),)
    )
    # Figure += geom_line(df_additional, aes(x=Axes_Name[0], y=Axes_Name[1]), color='tomato', size=line_size, alpha=1)
    try: 
       Figure.save(os.path.join(dir_pnet_QC, 'Delta_Spatial_Correspondence.jpg'), verbose=False, dpi=500)
    except:
       print("All subjects failed the QA")
    # # visualization for spatial correspondence
    # before = np.nanmean(Delta_Spatial_Correspondence)
    # after = np.nanmean(Spatial_Correspondence, axis=1)
    #
    # Axes_Name = ['Network Correspondence', 'Spatial Correspondence']
    # Group_Name = ['Unmatched', 'Matched']
    # Group_Color = ['tomato', 'dodgerblue']
    # Line_Color = 'gray'
    #
    # n = before.shape[0]
    #
    # df = pd.DataFrame({
    #     Axes_Name[1]: np.hstack([before, after]),
    #     Axes_Name[0]: np.repeat(Group_Name, n),
    #     'id': np.hstack([range(n), range(n)])
    # })
    #
    # df[Axes_Name[0]] = df[Axes_Name[0]].astype(pdtypes.CategoricalDtype(categories=Group_Name))
    # df.head()
    #
    # shift = 0.1
    #
    # def alt_sign(x):
    #     return (-1) ** x
    #
    # m1 = aes(x=stage(Axes_Name[0], after_scale='x+shift*alt_sign(x)'))  # shift outward
    # m2 = aes(x=stage(Axes_Name[0], after_scale='x-shift*alt_sign(x)'), group='id')  # shift inward
    #
    # line_size = 0.6
    # # set the transparency for filling area
    # fill_alpha = 0.8
    # point_alpha = 0.5
    # # set the transparency for lines
    # if n < 10:
    #     line_alpha = 1
    # elif n < 50:
    #     line_alpha = 0.5
    # elif n < 100:
    #     line_alpha = 0.3
    # elif n < 500:
    #     line_alpha = 0.2
    # elif n < 1000:
    #     line_alpha = 0.1
    # else:
    #     line_alpha = 0.05
    #
    # Figure = (ggplot(df, aes(Axes_Name[0], Axes_Name[1], fill=Axes_Name[0]))
    #           + geom_violin(m1, style='left-right', alpha=fill_alpha, size=line_size, show_legend=False)
    #           + geom_line(m2, color=Line_Color, size=line_size, alpha=line_alpha)
    #           + geom_point(m2, color='none', alpha=point_alpha, size=2, show_legend=False)
    #           + geom_boxplot(width=shift, alpha=fill_alpha, size=line_size, outlier_alpha=point_alpha,
    #                          show_legend=False)
    #           + scale_fill_manual(values=Group_Color)
    #           + theme_classic()
    #           + theme(figure_size=(4, 4),
    #                   axis_title=element_text(family='Arial', size=16, weight='bold', color='black'),
    #                   axis_text=element_text(family='Arial', size=14, weight='bold', color='black'),
    #                   axis_line=element_line(size=2, color='black'), )
    #           )
    #
    # Figure.save(os.path.join(dir_pnet_QC, 'Spatial_Correspondence.jpg'), verbose=False, dpi=500)

    # # visualization for spatial correspondence
    # # minimum of delta spatial correspondence
    # min_delta_SC = np.min(Delta_Spatial_Correspondence, axis=1)
    # min_delta_SC[np.isnan(min_delta_SC)] = 0
    # failed = pd.Series(min_delta_SC[min_delta_SC < 0], index=np.where(min_delta_SC < 0)[0])
    # passed = pd.Series(min_delta_SC[min_delta_SC >= 0], index=np.where(min_delta_SC >= 0)[0])
    #
    # Axes_Name = ['Quality Control', 'Delta Spatial Correspondence']
    # Group_Name = ['Failed', 'Passed']
    # Group_Color = ['tomato', 'dodgerblue']
    #
    # # Align the lengths of the two columns
    # max_len = max(len(failed), len(passed))
    # if len(failed) > 0:
    #     failed = failed.reindex(range(max_len)).fillna(0)
    #     all_pass = False
    # else:
    #     failed = pd.Series(np.full(max_len, np.NAN), index=range(max_len))
    #     all_pass = True
    # passed = passed.reindex(range(max_len)).fillna(0)
    #
    # # Create DataFrame
    # df = pd.DataFrame({
    #     Axes_Name[1]: np.hstack([failed.values, passed.values]),
    #     Axes_Name[0]: np.repeat(Group_Name, max_len),
    #     'id': np.hstack([range(max_len), range(max_len)])
    # })
    #
    # df[Axes_Name[0]] = df[Axes_Name[0]].astype(pdtypes.CategoricalDtype(categories=Group_Name))
    # df.head()
    #
    # shift = 0.1
    #
    # def alt_sign(x):
    #     return (-1) ** x
    #
    # if all_pass:
    #     m1 = aes(x=stage(Axes_Name[0], after_scale='x+shift*alt_sign(x)'))  # shift outward
    #     m2 = aes(x=stage(Axes_Name[0], after_scale='x-shift*alt_sign(x)'))  # shift inward
    #     violin_style = 'right'
    # else:
    #     m1 = aes(x=stage(Axes_Name[0], after_scale='x+shift*alt_sign(x)'))  # shift outward
    #     m2 = aes(x=stage(Axes_Name[0], after_scale='x-shift*alt_sign(x)'))  # shift inward
    #     violin_style = 'left-right'
    #
    # line_size = 0.6
    # # set the transparency for filling area
    # fill_alpha = 0.8
    # point_alpha = 0.5
    #
    # Figure = (ggplot(df, aes(Axes_Name[0], Axes_Name[1], fill=Axes_Name[0]))
    #           + geom_violin(m1, style=violin_style, alpha=fill_alpha, size=line_size, show_legend=False)
    #           + geom_point(m2, color='none', alpha=point_alpha, size=2, show_legend=False)
    #           + geom_boxplot(width=shift, alpha=fill_alpha, size=line_size, outlier_alpha=point_alpha,
    #                          show_legend=False)
    #           + scale_fill_manual(values=Group_Color)
    #           + theme_classic()
    #           + theme(figure_size=(4, 4),
    #                   axis_title=element_text(family='Arial', size=16, weight='bold', color='black'),
    #                   axis_text=element_text(family='Arial', size=14, weight='bold', color='black'),
    #                   axis_line=element_line(size=2, color='black'), )
    #           )
    #
    # Figure.save(os.path.join(dir_pnet_QC, 'Delta_Spatial_Correspondence.jpg'), verbose=False, dpi=500)

    # visualization for functional coherence
    before = np.nanmean(Functional_Coherence_Control[ps_passed, :], axis=1)
    after = np.nanmean(Functional_Coherence[ps_passed, :], axis=1)

    Axes_Name = ['Functional Network Definition', 'Functional Coherence']
    Group_Name = ['Group', 'Personalized']
    Group_Color = ['tomato', 'dodgerblue']
    Line_Color = 'gray'

    n = before.shape[0]

    df = pd.DataFrame({
        Axes_Name[1]: np.hstack([before, after]),
        Axes_Name[0]: np.repeat(Group_Name, n),
        'id': np.hstack([range(n), range(n)])
    })

    df[Axes_Name[0]] = df[Axes_Name[0]].astype(pdtypes.CategoricalDtype(categories=Group_Name))
    df.head()

    shift = 0.1

    def alt_sign(x):
        return (-1) ** x

    m1 = aes(x=stage(Axes_Name[0], after_scale='x+shift*alt_sign(x)'))              # shift outward
    m2 = aes(x=stage(Axes_Name[0], after_scale='x-shift*alt_sign(x)'), group='id')  # shift inward

    line_size = 0.6
    # set the transparency for filling area
    fill_alpha = 0.8
    point_alpha = 0.5
    # set the transparency for lines
    if n < 10:
        line_alpha = 1
    elif n < 50:
        line_alpha = 0.5
    elif n < 100:
        line_alpha = 0.3
    elif n < 500:
        line_alpha = 0.2
    elif n < 1000:
        line_alpha = 0.1
    else:
        line_alpha = 0.05

    Figure = (ggplot(df, aes(Axes_Name[0], Axes_Name[1], fill=Axes_Name[0]))
     + geom_violin(m1, style='left-right', alpha=fill_alpha, size=line_size, show_legend=False)
     + geom_line(m2, color=Line_Color, size=line_size, alpha=line_alpha)
     + geom_point(m2, color='none', alpha=point_alpha, size=2, show_legend=False)
     + geom_boxplot(width=shift, alpha=fill_alpha, size=line_size, outlier_alpha=point_alpha, show_legend=False)
     + scale_fill_manual(values=Group_Color)
     + theme_classic()
     + theme(figure_size=(4, 4),
             axis_title=element_text(family='Arial', size=16, weight='bold', color='black'),
             axis_text=element_text(family='Arial', size=14, weight='bold', color='black'),
             axis_line=element_line(size=2, color='black'),)
    )

    try:
      Figure.save(os.path.join(dir_pnet_QC, 'Functional_Coherence.jpg'), verbose=False, dpi=500)
    except:
      print("All subjects failed the QA")
