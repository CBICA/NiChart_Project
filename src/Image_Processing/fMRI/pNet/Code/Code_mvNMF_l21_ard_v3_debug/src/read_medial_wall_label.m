
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
