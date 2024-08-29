function sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath,timepointFile,svFileL,svFileR)
    % Modified by Yuncong Ma, 9/7/2022
    % add timepointFile

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

% get sbj list
disp('Get subject list...');
fid = fopen(sbjListFile,'r');
sbjList = textscan(fid,'%s');
fclose(fid);

if nargin==3
    % read data for each vertex
    disp('Extract data for each vertex...');
    svxOn = 0;
elseif nargin==5
    % read data for each super-vertex
    disp('Extract data for each super-veretex...');
    svxOn = 1;
    hemiSvFile = {svFileL,svFileR};
else
    error('number of input should be 3 or 5 !');
end

hemiSet = {'L','R'};

disp('read cifti data...');
sbjData = cell(length(sbjList{1}),1);
for si=1:length(sbjList{1})
    fileName = sbjList{1}{si};
    disp([num2str(si),'. ',fileName]);
    
    if exist(fileName,'file')
        cii = ciftiopen(fileName,wbPath);   % contain cii.cdata: vNum X tNum
    else
        disp('  this file does not exist !');
        continue;
    end
        
    
    if ~isempty(List_timepointFile)
        if ~isempty(List_timepointFile{si})
            load(List_timepointFile{si},'Time_Point');% Time_Point
            cii.cdata=cii.cdata(:,Time_Point);
        else
            cii.cdata=[];
        end
    end

    tmpData = [];
    for hi=1:length(hemiSet)
        hem = hemiSet{hi};
        
        if strcmp(hem, 'L')
            tmpTs = cii.cdata(1:29696,:);
        elseif strcmp(hem, 'R')
            tmpTs = cii.cdata(29697:59412,:);
        else
            tmpTs = cii.cdata(59413:end,:);
        end
        
        if svxOn==1 
            load(hemiSvFile{hi},'superLabels');
            uLab = unique(superLabels);
            uLabNum = length(uLab);
        
            tNum = size(tmpTs,2);
            tmpHmData = zeros(uLabNum,tNum,'single');
            for uli=1:uLabNum
                tmpHmData(uli,:) = mean(tmpTs(superLabels==uLab(uli),:));
            end
            
            tmpData = [tmpData;tmpHmData];
        else
            tmpData = [tmpData;tmpTs];
        end
        
        clear tmpTs;
    end
    clear cii;
    
    sbjData{si} = tmpData';
end

nonEmpty = ones(length(sbjData),1);
for si=1:length(sbjData)
    if isempty(sbjData{si})
        nonEmpty(si,1) = 0;
    end
end
sbjData = sbjData(logical(nonEmpty),1);

