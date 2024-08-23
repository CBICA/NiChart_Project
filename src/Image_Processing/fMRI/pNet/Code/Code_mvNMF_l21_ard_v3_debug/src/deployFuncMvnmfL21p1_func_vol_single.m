function deployFuncMvnmfL21p1_func_vol_single(sbjListFile,sbjTCFolder,maskFile,prepDataFile,outDir,resId,initName,K,alphaS21,alphaL,vxI,spaR,ard,eta,iterNum,calcGrp,parforOn)

if nargin~=17
    error('number of input should be 17 !');
end

if isdeployed
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


%sbjData = prepareFuncData_vol_func_2(sbjListFile,imgNumPerSbj,maskFile);   
if ~exist([sbjTCFolder, '/sbjData.mat'], 'file')
    sbjData = prepareFuncData_vol_func(sbjListFile,maskFile);
    if ~exist(sbjTCFolder,'dir')
        mkdir(sbjTCFolder);
    end
    save([sbjTCFolder, '/sbjData.mat'], 'sbjData', '-v7.3');
    %JobStatus = 'Preprocessing';
    %save([outDir, '/JobStatus.mat'], 'JobStatus');
else
    JobStatus = 'Preprocessing';
    save([outDir, '/JobStatus.mat'], 'JobStatus');
    load([sbjTCFolder, '/sbjData.mat']);
end

sbjNum = length(sbjData);
sbjNum

%if ~exist(outDir,'dir')
%    mkdir(outDir);
%end

K
parforOn

tic;
func_mMvNMF4fmri_l21p1_ard_woSrcLoad(sbjData,prepDataFile,outDir,resId,initName,sbjNum,K,alphaS21,alphaL,spaR,vxI,ard,eta,iterNum,calcGrp,parforOn);
toc;

if isdeployed
    exit;
else
    disp('Done!');
end
