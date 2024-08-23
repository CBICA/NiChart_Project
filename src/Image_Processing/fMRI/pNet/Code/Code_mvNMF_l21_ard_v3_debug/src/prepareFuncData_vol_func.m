function sbjData = prepareFuncData_vol_func(fileList,maskName)
% fileList -- path of functional files (.nii): each row for one image
% maskName -- path of the brain mask nii
%
% output
% sbjData contains a cell structure sbjData (cell(sbjNum,1)), in which each 
% entry sbjData{i} is a matrix of size t x v (# of time points by # of voxels)
%

if nargin~=2
    error('Usage: prepareFuncData_vol fileList maskName');
end

% read image list
fileID = fopen(fileList);
sbjList = textscan(fileID,'%s');
sbjList = sbjList{1};
fclose(fileID);

% load brain mask image
maskNii = load_untouch_nii(maskName);
maskMat = maskNii.img~=0;

sbjNum = length(sbjList);
sbjData = cell(sbjNum,1);

disp('Read images...');
for si=1:sbjNum
    disp([' ',num2str(si),'. ',sbjList{si}]);
    sbjNii = load_untouch_nii(sbjList{si});
    
    vxNum = sum(maskMat(:)~=0);
    tNum = size(sbjNii.img,4);
    dataMat = zeros(tNum,vxNum,'single');
    for ti=1:tNum
        tImg = sbjNii.img(:,:,:,ti);
        dataMat(ti,:) = tImg(maskMat);
    end
    sbjData{si} = dataMat;
end


