
# setup all sub-folders in the pnet result folder
dir_pnet_dataInput, dir_pnet_FNC, dir_pnet_gFN, dir_pnet_pFN, dir_pnet_QC, dir_pnet_STAT = pnet.setup_result_folder(dir_pnet_result)

# ============== Setup ============== #
# ============== Data Input
# setup dataInput
pnet.setup_scan_info(
    dir_pnet_dataInput=dir_pnet_dataInput,
    dataType=dataType, dataFormat=dataFormat,
    file_scan=file_scan, file_subject_ID=file_subject_ID,
    file_subject_folder=file_subject_folder, file_group_ID=file_group_ID,
    Combine_Scan=Combine_Scan
)
# setup brain template
# Volume and surface data types require different inputs to compute the brain template
if file_Brain_Template is None:
    if dataType == 'Volume':
        pnet.setup_brain_template(
            dir_pnet_dataInput=dir_pnet_dataInput,
            dataType=dataType,
            templateFormat=templateFormat,
            file_mask_vol=file_mask_vol, file_overlayImage=file_overlayImage,
            maskValue=maskValue
        )
    elif dataType == 'Surface':
        pnet.setup_brain_template(
            dir_pnet_dataInput=dir_pnet_dataInput,
            dataType=dataType,
            templateFormat=templateFormat,
            file_surfL=file_surfL, file_surfR=file_surfR,
            file_maskL=file_maskL, file_maskR=file_maskR,
            maskValue=maskValue,
            file_surfL_inflated=file_surfL_inflated, file_surfR_inflated=file_surfR_inflated
        )
    elif dataType == 'Surface-Volume':
        pnet.setup_brain_template(
            dir_pnet_dataInput=dir_pnet_dataInput,
            dataType=dataType, dataFormat=dataFormat,
            templateFormat=templateFormat,
            file_surfL=file_surfL, file_surfR=file_surfR,
            file_maskL=file_maskL, file_maskR=file_maskR,
            file_mask_vol=file_mask_vol, file_overlayImage=file_overlayImage,
            maskValue=maskValue,
            file_surfL_inflated=file_surfL_inflated, file_surfR_inflated=file_surfR_inflated
        )

else:
    pnet.setup_brain_template(dir_pnet_dataInput, file_Brain_Template)

# ============== FN Computation
if method == 'SR-NMF':
    pnet.SR_NMF.setup_SR_NMF(
        dir_pnet_result=dir_pnet_result,
        K=K,
        init=init,
        sampleSize=sampleSize,
        nBS=nBS,
        nTPoints=nTPoints,
        Combine_Scan=Combine_Scan,
        file_gFN=file_gFN,
        Computation_Mode=Computation_Mode,
        dataPrecision=dataPrecision,
        outputFormat=outputFormat
    )
    if FN_model_parameter not in locals():
        FN_model_parameter = None
    if FN_model_parameter is not None:
        pnet.SR_NMF.update_model_parameter(dir_pnet_result, FN_model_parameter=FN_model_parameter)

elif method == 'GIG-ICA':
    pnet.GIG_ICA.setup_GIG_ICA(
        dir_pnet_result=dir_pnet_result,
        K=K,
        Combine_Scan=Combine_Scan,
        file_gFN=file_gFN,
        Computation_Mode=Computation_Mode,
        dataPrecision=dataPrecision,
        outputFormat=outputFormat
    )
    if FN_model_parameter is not None:
        pnet.GIG_ICA.update_model_parameter(dir_pnet_result, FN_model_parameter=FN_model_parameter)

# =============== Visualization
pnet.setup_Visualization(
    dir_pnet_result=dir_pnet_result,
    synchronized_view=synchronized_view,
    synchronized_colorbar=synchronized_colorbar
)

# =============== Cluster
pnet.setup_cluster(
    dir_env=dir_env,
    dir_pnet=dir_pnet,
    dir_pnet_result=dir_pnet_result,
    dir_python=dir_python,
    submit_command=submit_command,
    thread_command=thread_command,
    memory_command=memory_command,
    log_command=log_command,
    computation_resource=computation_resource
)

print('All setups are finished\n', flush=True)

# ============== Run ============== #
print('Start to run\n', flush=True)
# FN Computation
pnet.run_FN_computation_torch_cluster(dir_pnet_result)

# Quality Control
pnet.run_quality_control_torch_cluster(dir_pnet_result)

# Visualization
pnet.run_Visualization_cluster(dir_pnet_result)

# Web Report
pnet.run_web_report(dir_pnet_result)

print('All runs are finished\n', flush=True)