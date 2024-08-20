# Yuncong Ma, 2/14/2024
# Make a web page based report for fast visual examination

from Module.Visualization import *
import shutil


dir_python = os.path.dirname(os.path.abspath(__file__))


def run_web_report(dir_pnet_result: str):
    """
    generate HTML based web report

    :param dir_pnet_result:
    :return:

    Yuncong Ma, 2/14/2024
    """

    # get directories of sub-folders
    dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, _ = setup_result_folder(dir_pnet_result)

    # log file
    logFile = os.path.join(dir_pnet_result, 'Log_Report.log')
    logFile = open(logFile, 'w')
    print_log('\nStart generating HTML based web report at ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n',
              logFile=logFile, stop=False)

    # load settings for data input and FN computation
    if not os.path.isfile(os.path.join(dir_pnet_dataInput, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder Data_Input')
    if not os.path.isfile(os.path.join(dir_pnet_FNC, 'Setting.json')):
        raise ValueError('Cannot find the setting json file in folder FN_Computation')
    settingDataInput = load_json_setting(os.path.join(dir_pnet_dataInput, 'Setting.json'))
    settingFNC = load_json_setting(os.path.join(dir_pnet_FNC, 'Setting.json'))
    setting = {'Data_Input': settingDataInput, 'FN_Computation': settingFNC}
    print_log('Settings are loaded from folder Data_Input and FN_Computation', logFile=logFile, stop=False)

    # load basic settings
    dataType = setting['Data_Input']['Data_Type']
    dataFormat = setting['Data_Input']['Data_Format']
    Combine_Scan = setting['Data_Input']['Combine_Scan']

    # info about the fMRI dataset
    file_scan = os.path.join(dir_pnet_dataInput, 'Scan_List.txt')
    file_subject_ID = os.path.join(dir_pnet_dataInput, 'Subject_ID.txt')
    file_subject_folder = os.path.join(dir_pnet_dataInput, 'Subject_Folder.txt')

    list_scan = load_txt_list(file_scan)
    nScan = len(list_scan)
    list_subject_ID = load_txt_list(file_subject_ID)
    nSubject = len(np.unique(list_subject_ID))
    list_subject, subject_index = np.unique(load_txt_list(file_subject_ID), return_index=True)
    list_subject_ID_unqiue = list_subject_ID[subject_index]
    list_subject_folder = load_txt_list(file_subject_folder)
    list_subject_folder_unique, folder_index = np.unique(list_subject_folder, return_index=True)
    nFolder = len(list_subject_folder_unique)

    # =========== Summary =========== #
    # template for web page
    template_summary = os.path.join(dir_python, 'Web_Template_Summary.html')

    # Generate the summary web page
    file_summary = os.path.join(dir_pnet_result, 'Report.html')
    pnet_FN_method = setting['FN_Computation']['Method']
    K = setting['FN_Computation']['K']

    with open(template_summary, 'r') as file:
        html_as_string = file.read()
    # report title
    report_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    html_as_string = html_as_string.replace('{$report_time$}', str(report_time))
    # setting
    html_as_string = html_as_string.replace('{$pnet_FN_method$}', str(pnet_FN_method))
    html_as_string = html_as_string.replace('{$K$}', str(K))
    html_as_string = html_as_string.replace('{$dataType$}', str(dataType))
    html_as_string = html_as_string.replace('{$dataFormat$}', str(dataFormat))
    html_as_string = html_as_string.replace('{$nScan$}', str(nScan))
    html_as_string = html_as_string.replace('{$nSubject$}', str(nSubject))
    # Compress images if not done yet
    image_size = (2000, 10000)
    if not os.path.isfile(os.path.join(dir_pnet_gFN, 'All(Compressed).jpg')):
        compress_image(os.path.join(dir_pnet_gFN, 'All(Compressed).jpg'),
                       os.path.join(dir_pnet_gFN, 'All.jpg'),
                       image_size=image_size)
    for i in range(nFolder):
        if os.path.isfile(os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'All.jpg')) and not os.path.isfile(os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'All(Compressed).jpg')):
            compress_image(os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'All(Compressed).jpg'),
                           os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'All.jpg'),
                           image_size=image_size)

    # gFN
    if setting['FN_Computation']['Group_FN']['file_gFN'] is None:
        text_gFN = 'The group FNs are derived using the whole fMRI dataset'
    else:
        text_gFN = 'The group FNs are loaded from precomputed results at ' + setting['FN_Computation']['Group_FN']['file_gFN']
    html_as_string = html_as_string.replace('{$text_gFN$}', str(text_gFN))
    figure_gFN = './Group_FN/All(Compressed).jpg'
    html_as_string = html_as_string.replace('{$figure_gFN$}', str(figure_gFN))
    # pFN examples
    nMax = 10
    frames = [Image.open(os.path.join(dir_pnet_pFN, list_subject_folder[subject_index[i]], 'All(Compressed).jpg')) for i in range(np.minimum(nMax, nSubject))]
    frame_one = frames[0]
    frame_one.save(os.path.join(dir_pnet_pFN, 'Example.gif'), format="GIF", append_images=frames, save_all=True, duration=1000, loop=0, optimize=False)
    pFN_example = './' + os.path.join('Personalized_FN', 'Example.gif')
    html_as_string = html_as_string.replace('{$pFN_example$}', str(pFN_example))
    # pFN links
    link_pFN = ''
    pre_sub = list_subject_ID_unqiue[0]
    for i in range(nFolder):
        if list_subject_ID[folder_index[i]] != pre_sub:
            link_pFN = link_pFN + "<br />"
            pre_sub = list_subject_ID[folder_index[i]]
        file_pFN_indv = './' + os.path.join('Personalized_FN', list_subject_folder_unique[i], 'Report.html')
        link_pFN = link_pFN + f" <a href='{file_pFN_indv}' target='_blank' title='{list_subject_folder_unique[i]}'>({list_subject_folder_unique[i]})</a>\n"
    html_as_string = html_as_string.replace('{$link_pFN$}', str(link_pFN))
    # QC
    Result = load_matlab_single_variable(os.path.join(dir_pnet_QC, 'Result.mat'))
    n_pass = np.sum(np.min(Result['Delta_Spatial_Correspondence'][0, 0], axis=1) >= 0)
    n_missmatch = np.sum(np.min(Result['Delta_Spatial_Correspondence'][0, 0], axis=1) < 0)
    ps_missmatch = np.where(np.min(Result['Delta_Spatial_Correspondence'][0, 0], axis=1) < 0)[0]
    scan_subject = ''
    if Combine_Scan:
        scan_subject = 'subjects'
    else:
        scan_subject = 'scans'
    if n_missmatch == 0:
        text_qc = 'pFNs of all '+str(n_pass+n_missmatch)+' '+scan_subject+' passed QC. <br />'
    else:
        text_qc = 'There are ' + str(n_pass) + ' out of ' + str(n_pass+n_missmatch) + ' '+scan_subject+' passed QC. <br />\n' +\
                  'There are ' + str(n_missmatch) +' scans do not pass QC, meaning that they have at least one pFN showing smaller similarity to their group-level counterpart. <br />\n'
        text_qc += 'Below are the individual reports of '+scan_subject+' that do not pass QC. <br />\n'
        for i, ps in enumerate(ps_missmatch):
            file_pFN_indv = './' + os.path.join('Personalized_FN', list_subject_folder_unique[ps], 'Report.html')
            text_qc = text_qc + f" <a href='{file_pFN_indv}' target='_blank' title='{list_subject_folder_unique[ps]}'>({list_subject_folder_unique[ps]})</a>\n"
            if (i+1) % 10 == 0:
                text_qc = text_qc + " <br />"
    html_as_string = html_as_string.replace('{$text_qc$}', str(text_qc))

    file_summary = open(file_summary, 'w')
    print(html_as_string, file=file_summary)
    file_summary.close()

    # =========== Individual =========== #
    # template for web page
    template_individual = os.path.join(dir_python, 'Web_Template_Individual.html')
    pre_sub = ''
    delta_SC = Result['Delta_Spatial_Correspondence'][0, 0]
    for i in range(nFolder):
        if list_subject_ID[folder_index[i]] != pre_sub:
            link_pFN = link_pFN + "<br />"
            pre_sub = list_subject_ID[folder_index[i]]
            with open(template_individual, 'r') as file:
                html_as_string = file.read()
            # copy gFN figure
            shutil.copyfile(os.path.join(dir_pnet_gFN, 'All(Compressed).jpg'), os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'gFN_All(Compressed).jpg'))
            # report title
            html_as_string = html_as_string.replace('{$subject_info$}', str(pre_sub))
            report_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            html_as_string = html_as_string.replace('{$report_time$}', str(report_time))
            # setting
            html_as_string = html_as_string.replace('{$pnet_FN_method$}', str(pnet_FN_method))
            html_as_string = html_as_string.replace('{$K$}', str(K))
            html_as_string = html_as_string.replace('{$dataType$}', str(dataType))
            html_as_string = html_as_string.replace('{$dataFormat$}', str(dataFormat))
            html_as_string = html_as_string.replace('{$nScan$}', str(nScan))
            html_as_string = html_as_string.replace('{$nSubject$}', str(nSubject))
            html_as_string = html_as_string.replace('{$text_gFN$}', str(text_gFN))

            if np.sum(delta_SC[i, :] < 0) == 0:
                text_qc = 'All pFNs passed QC. All of them show higher spatial similarity to their group-level counterparts that others.'
            else:
                text_qc = f'This pFN result violates QC, meaning that some pFNs show lower spatial similarity to their group-level counterparts that others. <br /> \n' \
                          f'<br />\nDetails are below. <br />\n'
                Result = load_matlab_single_variable(os.path.join(dir_pnet_QC, list_subject_folder_unique[i], 'Result.mat'))
                Miss_Match = Result['Miss_Match'][0, 0]
                for j in range(Miss_Match.shape[0]):
                    try:
                      text_qc += f'pFN {Miss_Match[j, 0]} is more similar to gFN {Miss_Match[j, 1]} <br />\n'
                    except:
                      print('the number of Miss Matched is ' + str(Miss_Match.shape[0]) + '\n')

            html_as_string = html_as_string.replace('{$text_qc$}', str(text_qc))

            file_individual = os.path.join(dir_pnet_pFN, list_subject_folder_unique[i], 'Report.html')
            file_individual = open(file_individual, 'w')
            print(html_as_string, file=file_individual)
            file_individual.close()
