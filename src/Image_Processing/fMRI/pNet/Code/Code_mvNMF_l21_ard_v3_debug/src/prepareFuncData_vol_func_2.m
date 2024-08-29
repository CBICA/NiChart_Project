function sbjData = prepareFuncData_vol_func_2(fileList, imgNumPerSbj, maskName)
% fileList -- path of functional files (.nii or .nii.gz): each row for one image,
%			  each subject have imgNumPerSbj images
% maskName -- path of the brain mask nii
%
% output
% sbjData contains a cell structure sbjData (cell(sbjNum,1)), in which each 
% entry sbjData{i} is a matrix of size t x v (# of time points by # of voxels)
%

if nargin~=3
    error('Usage: prepareFuncData_vol fileList imgNumPerSbj maskName');
end

% read image list
fileID = fopen(fileList);
sbjList = textscan(fileID,'%s');
sbjList = sbjList{1};
fclose(fileID);

% load brain mask image
maskNii = load_untouch_nii(maskName);
maskMat = maskNii.img~=0;
vxNum = sum(maskMat(:)~=0);

sbjNum = length(sbjList) / imgNumPerSbj;
if sbjNum~=round(sbjNum)
	error('input fileList not valid, some subjects do not have imgNumPerSbj images');
end

sbjData = cell(sbjNum,1);

disp('Read images...');
for si=1:sbjNum
	disp([num2str(si), ': ']);

	dataMat = [];
	for ii=1:imgNumPerSbj
		ii_ind = (si-1)*imgNumPerSbj + ii;
		ii_file = sbjList{ii_ind};

	    disp(['  ', num2str(ii), '. ', ii_file]);
    	sbjNii = load_untouch_nii(ii_file);
    
    	tNum = size(sbjNii.img, 4);
    	ii_dataMat = zeros(tNum,vxNum,'single');
    	for ti=1:tNum
        	tImg = sbjNii.img(:,:,:,ti);
        	ii_dataMat(ti,:) = tImg(maskMat);
    	end

    	dataMat = [dataMat; ii_dataMat];
    end
    sbjData{si} = dataMat;

    total_t_num = size(sbjData{si},1);
    disp(['  total number of time points: ', num2str(total_t_num)]);
end


