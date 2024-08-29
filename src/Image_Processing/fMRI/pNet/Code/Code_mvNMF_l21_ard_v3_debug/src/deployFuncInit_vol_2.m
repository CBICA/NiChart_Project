function deployFuncInit_vol_2(sbjListFile,imgNumPerSbj,maskFile,prepDataFile,outDir,spaR,vxI,ard,iterNum,K,tNum,alpha,beta,resId)

if nargin~=14
    error('number of input should be 14 !');
end
    
if isdeployed
    spaR = str2double(spaR);
    vxI = str2double(vxI);
    ard = str2double(ard);
    iterNum = str2double(iterNum);
    K = str2double(K);
    tNum = str2double(tNum);
    alpha = str2double(alpha);
    beta = str2double(beta);
    imgNumPerSbj = str2double(imgNumPerSbj);
end

if ~exist(prepDataFile,'file')
    maskNii = load_untouch_nii(maskFile);
    maskMat = int32(maskNii.img~=0);
    gNb = constructW_vol(maskMat,spaR);

    save(prepDataFile,'gNb','-v7.3');
else
    load(prepDataFile); % containing gNb
end

nmVec = zeros(length(gNb),1);
for gni=1:length(gNb)
    nmVec(gni) = length(gNb{gni});
end
nM = median(nmVec);

sbjData = prepareFuncData_vol_func_2(sbjListFile, imgNumPerSbj, maskFile);    

%in_tNum = size(sbjData{1},1);
%if tNum~=in_tNum
%    error('tNum is not set correctly (not consistent with 1st dimension of input data), please double check');
%end

numUsed = length(sbjData);
pS = round((alpha*tNum*numUsed)/K);
pL = round((beta*tNum*numUsed)/(K*nM));

tic;
func_initialization_woLoadSrc(sbjData,prepDataFile,outDir,resId,numUsed,K,pS,pL,spaR,vxI,ard,iterNum);
toc;

if isdeployed
    exit;
end
