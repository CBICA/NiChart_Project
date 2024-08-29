function sbjData = prepareFuncData_mat_func(fileList,maskFile)
% fileList -- path of functional files (.mat): each row for one subject, each .mat file contains
%			  a variable named data (a matrix with size t x v, # of time points by # of voxels before masking) 
% maskFile -- mask data file, containing vectorized mask, named data
%
% output
% sbjData contains a cell structure sbjData (cell(sbjNum,1)), in which each 
% entry sbjData{i} is a matrix of size t x v (# of time points by # of voxels)
%

if nargin~=2
    error('Usage: prepareFuncData_mat_func fileList maskFile');
end

% read image list
fileID = fopen(fileList);
sbjList = textscan(fileID,'%s');
sbjList = sbjList{1};
fclose(fileID);

% load brain mask image
maskMat = load(maskFile);
maskVec = maskMat.data > 0;

%maskNii = load_untouch_nii(maskImageFile);
%maskVec = maskNii.img(:) > 0;

sbjNum = length(sbjList);
sbjData = cell(sbjNum,1);

disp('Read data...');
for si=1:sbjNum
    disp([' ',num2str(si),'. ',sbjList{si}]);
    sbjMat = load(sbjList{si});
    
    sbjData{si} = sbjMat.data(:,maskVec);
end


