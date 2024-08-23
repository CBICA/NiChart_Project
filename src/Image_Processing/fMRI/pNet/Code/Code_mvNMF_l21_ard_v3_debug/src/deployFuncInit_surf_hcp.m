function deployFuncInit_surf_hcp(sbjListFile,wbPath,surfL,surfR,surfML,surfMR,prepDataFile,outDir,spaR,vxI,ard,iterNum,K,tNum,alpha,beta,resId,svFileL,svFileR)

if nargin~=17 && nargin~=19
    error('number of input should be 13 or 15 !');
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
    [surfStru, surfMask] = getHcpSurf(surfL, surfR, surfML, surfMR);
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

if nargin==17
    sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath);    
elseif nargin==19
    sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath,svFileL,svFileR);
end

numUsed = length(sbjData);
pS = round((alpha*tNum*numUsed)/K);
pL = round((beta*tNum*numUsed)/(K*nM));

tic;
func_initialization_woLoadSrc(sbjData,prepDataFile,outDir,resId,numUsed,K,pS,pL,spaR,vxI,ard,iterNum);
toc;

if isdeployed
    exit;
end
