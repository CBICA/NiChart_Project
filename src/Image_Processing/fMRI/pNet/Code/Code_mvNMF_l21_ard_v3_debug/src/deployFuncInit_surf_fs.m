function deployFuncInit_surf_fs(sbjListFile,surfL,surfR,surfML,surfMR,prepDataFile,outDir,spaR,vxI,ard,iterNum,K,tNum,alpha,beta,resId)

if nargin~=16
    error('number of input should be 16 !');
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
end

if ~exist(prepDataFile,'file')
    [surfStru, surfMask] = getFsSurf(surfL, surfR, surfML, surfMR);
    gNb = constructW_surf(surfStru, spaR, surfMask);

    save(prepDataFile,'gNb','-v7.3');
else
    load(prepDataFile); % containing gNb
end

nmVec = zeros(length(gNb),1);
for gni=1:length(gNb)
    nmVec(gni) = length(gNb{gni});
end
nM = median(nmVec);

sbjData = prepareFuncData_fs_func(sbjListFile, surfML, surfMR);

numUsed = length(sbjData);
pS = round((alpha*tNum*numUsed)/K);
pL = round((beta*tNum*numUsed)/(K*nM));

tic;
func_initialization_woLoadSrc(sbjData,prepDataFile,outDir,resId,numUsed,K,pS,pL,spaR,vxI,ard,iterNum);
toc;

if isdeployed
    exit;
end
