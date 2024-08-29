function deployFuncInit_surf_hcp_tp(sbjListFile,wbPath,surfL,surfR,surfML,surfMR,prepDataFile,outDir,sublist_id,spaR,vxI,ard,iterNum,K,tNum,alpha,beta,resId,timepointFile,svFileL,svFileR)
% Modified by Yuncong Ma, 9/7/2022
% Add timepointFile

Debug=1;
if Debug==1
    fprintf('\n-- deployFuncInit_surf_hcp_tp --\n');
end

if nargin~=19 && nargin~=21
    error('number of input should be 19 or 21 !');
end
    
if isdeployed || ischar(spaR)
    sublist_id = str2double(sublist_id);
    spaR = str2double(spaR);
    vxI = str2double(vxI);
    ard = str2double(ard);
    iterNum = str2double(iterNum);
    K = str2double(K);
    tNum = str2double(tNum);
    alpha = str2double(alpha);
    beta = str2double(beta);
end

if Debug==1
    display(sbjListFile)
    display(wbPath)
    display(surfL)
    display(surfR)
    display(surfML)
    display(surfMR)
    display(prepDataFile);
    display(outDir);
    display(sublist_id);
    display(spaR);
    display(vxI);
    display(ard);
    display(iterNum);
    display(K);
    display(tNum);
    display(alpha);
    display(beta);
    display(resId);
    display(timepointFile);
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


if nargin==19
    sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath,timepointFile);    
elseif nargin==21
    sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath,timepointFile,svFileL,svFileR);
end

if Debug==1
    fprintf('\nData size is [%d,%d]\n',size(sbjData,1),size(sbjData,2));
end

numUsed = length(sbjData);
pS = round((alpha*tNum*numUsed)/K);
pL = round((beta*tNum*numUsed)/(K*nM));

tic;
func_initialization_woLoadSrc(sbjData,prepDataFile,outDir,sublist_id,resId,numUsed,K,pS,pL,spaR,vxI,ard,iterNum);
toc;

if isdeployed
    exit;
end
