function sbjData = prepareFuncData_fs_func(sbjListFile, timepointFile, medialWallFile_l, medialWallFile_r)
% sbjListFile: text file contains the path to image files for each subject,
%              there are two lines for each subject, one (1st) for left hemisphere, and one (2nd) for right
% 
% Modified by Yuncong Ma, 9/19/2022
% Add support for timepointFile

if exist(timepointFile,'file')
    FID=fopen(timepointFile);
    List_timepointFile=textscan(FID,'%s\n');
    List_timepointFile=List_timepointFile{1};
    fclose(FID);
elseif isempty(timepointFile) || strcmp(timepointFile,'0')
    List_timepointFile=[]; 
else
    error('prepareFuncData_hcp_func: Wrong input for timepointFile: %s\n',timepointFile);
end

if (nargin~=1) && (nargin~=4)
    error('There should be 1 or 4 input arguments.');
end

if nargin==4
    rmMedWall = 1;

    mwIndVec_l = read_medial_wall_label(medialWallFile_l);
    mwIndVec_r = read_medial_wall_label(medialWallFile_r);
end

% get sbj list
disp('Get subject list...');
fid = fopen(sbjListFile,'r');
sbjList = textscan(fid,'%s');
sbjList = sbjList{1};
fclose(fid);

%hemiSet = {'L','R'};

numSbj = length(sbjList) / 2;

disp('read mgz data...');
sbjData = cell(numSbj,1);

for si=1:numSbj
    fileName_l = sbjList{(si-1)*2+1};
    fileName_r = sbjList{si*2};

    disp([num2str(si),' l. ',fileName_l]);
    disp([num2str(si),' r. ',fileName_r]);
    
    if exist(fileName_l,'file') && exist(fileName_r,'file')
        mri_l = MRIread(fileName_l);
        mri_r = MRIread(fileName_r);

        data_l = squeeze(mri_l.vol);
        data_r = squeeze(mri_r.vol);
        
        if rmMedWall==1
            data_l(mwIndVec_l,:) = [];
            data_r(mwIndVec_r,:) = [];
        end

        tmpData = [data_l; data_r];     % v x t

        if ~isempty(List_timepointFile)
            if ~isempty(List_timepointFile{si})
                load(List_timepointFile{si},'Time_Point');% Time_Point
                tmpData=tmpData(:,Time_Point);
            else
                tmpData=[];
            end
        end
    else
        disp('  files does not exist !');
        continue;
    end
    
    sbjData{si} = tmpData';
end

nonEmpty = ones(length(sbjData),1);
for si=1:length(sbjData)
    if isempty(sbjData{si})
        nonEmpty(si,1) = 0;
    end
end
sbjData = sbjData(logical(nonEmpty),1);

end % end function


function mwInd = read_medial_wall_label(medialWallFile)
    lfid = fopen(medialWallFile,'r');
    fgets(lfid);    % pass the 1st line
    line = fgets(lfid);
    nv = sscanf(line, '%d');
    l = fscanf(lfid, '%d %f %f %f %f\n');
    l = reshape(l, 5, nv);
    l = l';
    fclose(lfid);

    mwInd = l(:,1) + 1; % note the vertex number is 0-based
end

