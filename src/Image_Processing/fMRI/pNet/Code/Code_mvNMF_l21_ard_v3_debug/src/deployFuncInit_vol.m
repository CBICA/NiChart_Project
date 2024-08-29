function deployFuncInit_vol(sbjListFile,maskFile,prepDataFile,outDir,sublist_id,spaR,vxI,ard,iterNum,K,tNum,alpha,beta,resId)

if nargin~=14
    error('number of input should be 14 !');
end
    
% if isdeployed
    sublist_id = str2double(sublist_id);
    spaR = str2double(spaR);
    vxI = str2double(vxI);
    ard = str2double(ard);
    iterNum = str2double(iterNum);
    K = str2double(K);
    tNum = str2double(tNum);
    alpha = str2double(alpha);
    beta = str2double(beta);
% end

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

sbjData = prepareFuncData_vol_func(sbjListFile,maskFile);    

numUsed = length(sbjData);



pS = round((alpha*tNum*numUsed)/K);
pL = round((beta*tNum*numUsed)/(K*nM));

tic;
func_initialization_woLoadSrc(sbjData,prepDataFile,outDir,sublist_id,resId,numUsed,K,pS,pL,spaR,vxI,ard,iterNum);
toc;

if isdeployed
    exit;
end
