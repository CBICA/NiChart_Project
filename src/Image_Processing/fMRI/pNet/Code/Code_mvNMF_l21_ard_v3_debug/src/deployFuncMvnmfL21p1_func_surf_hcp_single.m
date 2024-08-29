function deployFuncMvnmfL21p1_func_surf_hcp_single(sbjListFile,sbjTCFolder,wbPath,prepDataFile,outDir,resId,initName,K,alphaS21,alphaL,vxI,spaR,ard,eta,iterNum,calcGrp,parforOn)
% Yuncong Ma, 8/24/2022
% Modified from deployFuncMvnmfL21p1_func_surf_hcp
% Following Zheng Zhou's modification
% deployFuncMvnmfL21p1_func_vol_single.m
% Input only includes one data file
% 

if nargin~=17
    error('number of input should be 17 !');
end

if isdeployed || ischar(K)
    K = str2double(K);
    alphaS21 = str2double(alphaS21);
    alphaL = str2double(alphaL);
    vxI = str2double(vxI);
    spaR = str2double(spaR);
    ard = str2double(ard);
    eta = str2double(eta);
    iterNum = str2double(iterNum);
    calcGrp = str2double(calcGrp);
    parforOn = str2double(parforOn);
end

Debug=1;
if Debug==1
    fprintf('\n--deployFuncMvnmfL21p1_func_surf_hcp_single--\n');
    display(sbjListFile);
    display(sbjTCFolder);
    display(wbPath);
    display(prepDataFile);
    display(outDir);
    display(resId);
    display(initName);
    display(K);
    display(alphaS21);
    display(alphaL);
    display(vxI);
    display(spaR);
    display(ard);
    display(eta);
    display(iterNum);
    display(calcGrp);
    display(parforOn);
end

if ~exist([sbjTCFolder, '/sbjData.mat'], 'file')
    sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath);
    
    if ~exist(sbjTCFolder,'dir')
        mkdir(sbjTCFolder);
    end
%     save([sbjTCFolder, '/sbjData.mat'], 'sbjData', '-v7.3');
    JobStatus = 'Preprocessing';
    save([outDir, '/JobStatus.mat'], 'JobStatus');
else
    JobStatus = 'Preprocessing';
    if ~exist(outDir,'dir')
        mkdir(outDir);
    end
    save([outDir, '/JobStatus.mat'], 'JobStatus');
    load([sbjTCFolder, '/sbjData.mat']);
end

sbjNum = length(sbjData);
if Debug==1
    display(sbjNum);
end

tic;
func_mMvNMF4fmri_l21p1_ard_woSrcLoad(sbjData,prepDataFile,outDir,resId,initName,sbjNum,K,...
    alphaS21,alphaL,spaR,vxI,ard,eta,iterNum,calcGrp,parforOn);
toc;

if isdeployed
    exit;
else
    disp('Done!');
end
