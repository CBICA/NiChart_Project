function [surfStru, surfMask] = getFsSurf(surfNameL, surfNameR, surfMaskNameL, surfMaskNameR)
% prep freesurfer surface structure
% for example, surf file
% surfNameL = [fs_path,filesep,'subjects',filesep,'fsaverage5',filesep,'surf',filesep,'lh.pial'];
% surfNameR = [fs_path,filesep,'subjects',filesep,'fsaverage5',filesep,'surf',filesep,'rh.pial'];
% surfMaskNameL = [fs_path,filesep,'subjects',filesep,'fsaverage5',filesep,'label',filesep,'lh.Medial_wall.label'];
% surfMaskNameR = [fs_path,filesep,'subjects',filesep,'fsaverage5',filesep,'label',filesep,'rh.Medial_wall.label'];

% surface topology
[vx_l, faces_l] = read_surf(surfNameL);
[vx_r, faces_r] = read_surf(surfNameR);

surfStru.vx_l = vx_l;
surfStru.faces_l = faces_l + 1;
surfStru.vx_r = vx_r;
surfStru.faces_r = faces_r + 1;

% surface mask
l_l = read_label([],surfMaskNameL);
l_r = read_label([],surfMaskNameR);

l_l_ind = l_l(:,1) + 1;
l_r_ind = l_r(:,1) + 1;

surfMask.l = ones(length(vx_l),1);
surfMask.l(l_l_ind) = 0;

surfMask.r = ones(length(vx_r),1);
surfMask.r(l_r_ind) = 0;
