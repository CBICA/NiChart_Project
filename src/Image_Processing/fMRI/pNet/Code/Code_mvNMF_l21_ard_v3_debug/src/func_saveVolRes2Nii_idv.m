function func_saveVolRes2Nii_idv(resFileName,maskName,sbjId,outDir,saveFig,refNiiName)

if nargin==4
    saveFig = 0;
end

if nargin~=4 && nargin~=6
    error('Usage: func_saveVolRes2Nii_idv( resFileName,maskName,sbjId,outDir,(saveFig,refNiiName) )');
end

ld_res = load(resFileName);
allRes = ld_res.V;
clear V;

maskNii = load_untouch_nii(maskName);

if sbjId<10
    sbjIdStr = ['00',num2str(sbjId)];
elseif ki<100
    sbjIdStr = ['0',num2str(sbjId)];
else
    sbjIdStr = num2str(sbjId);
end

if ~exist(outDir,'dir')
    mkdir(outDir);
end

res = allRes{sbjId};
smInd = res ./ max(repmat(max(res),size(res,1),1),eps) < 1e-2;
res(smInd) = 0;

K = size(res,2);
for ki=1:K
    if ki<10
        kStr = ['00',num2str(ki)];
    elseif ki<100
        kStr = ['0',num2str(ki)];
    else
        kStr = num2str(ki);
    end
    disp(['save nii -- icn ',kStr]);
    
    kNii = maskNii;
    kNii.img(maskNii.img~=0) = res(:,ki);
    
    outName = [outDir,filesep,'icn_',kStr,'.nii.gz'];
    save_untouch_nii(kNii,outName);
end

if saveFig==1
    for ki=1:K
        if ki<10
            kStr = ['00',num2str(ki)];
        elseif ki<100
            kStr = ['0',num2str(ki)];
        else
            kStr = num2str(ki);
        end
        disp(['save fig -- icn ',kStr]);
        
        smNiiName = [outDir,filesep,'icn_',kStr,'.nii.gz'];
        outFigName = [outDir,filesep,'cutoff_icn_',kStr,'.tif'];

        dispSM_func(smNiiName,refNiiName,outFigName);
    end
end
