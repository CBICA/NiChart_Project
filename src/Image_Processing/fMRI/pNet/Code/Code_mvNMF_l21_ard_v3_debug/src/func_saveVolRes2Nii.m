function func_saveVolRes2Nii(resFileName,maskName,outDir,saveFig,refNiiName)

if nargin==3
    saveFig = 0;
end

if nargin~=3 && nargin~=5
    error('Usage: func_saveVolRes2Nii( resFileName,maskName,outDir,(saveFig,refNiiName) )');
end

res = load(resFileName);

if ~isfield(res, 'initV')
    initV = res.V_centroid;
else
    initV = res.initV;
end

maskNii = load_untouch_nii(maskName);

if ~exist(outDir,'dir')
    mkdir(outDir);
end

smInd = initV ./ max(repmat(max(initV),size(initV,1),1),eps) < 1e-2;
initV(smInd) = 0;

K = size(initV,2);
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
    kNii.img(maskNii.img~=0) = initV(:,ki);
    
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
