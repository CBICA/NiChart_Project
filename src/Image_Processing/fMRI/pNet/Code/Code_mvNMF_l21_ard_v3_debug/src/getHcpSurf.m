function [surfStru, surfMask] = getHcpSurf(surfNameL, surfNameR, surfMaskNameL, surfMaskNameR)
% prep hcp surface structure

%gdPath = 'C:\Users\LiHon\Google Drive';
%giftiPath = [gdPath,filesep,'Code\Download',filesep,'gifti-1.6'];
%addpath(genpath(giftiPath));

% surface topology
g_l = gifti(surfNameL);
g_r = gifti(surfNameR);

surfStru.faces_l = g_l.faces;
surfStru.faces_r = g_r.faces;
surfStru.vx_l = g_l.vertices;
surfStru.vx_r = g_r.vertices;

% surface mask
cm_l = gifti(surfMaskNameL);
cm_r = gifti(surfMaskNameR);

surfMask.l = cm_l.cdata > 0;
surfMask.r = cm_r.cdata > 0;
