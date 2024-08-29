function gNb = createPrepData(dataType, maskStru, nb, auxMask)
% create spatial neighborhood for volumetric or surface data
% Input:
% 	dataType: 'volumetric' or 'surface'
%	maskStru: if dataType is 'volumetric', maskStru should be matrix containing a mask image
%			  if dataType is 'surface', maskStru should be a surface structure including
%			        maskStru.vx_l: vertices of left hemi-sphere
%       			maskStru.vx_r: vertices of right hemi-sphere
%       			maskStru.faces_l: faces of left hemi-sphere
%       			maskStru.faces_r: faces of right hemi-sphere
% 	nb: size of spaital neighborhood
% 	auxMask: (required if dataType is 'surface')
%			a structure containing the mask of regions of interest (ROI) for each hemi-sphere, including
%           auxMask.l (vector) and auxMask.r (vector)

if (strcmp(dataType, 'volumetric') && nargin==3) || (strcmp(dataType, 'surface') && nargin==4)
	if strcmp(dataType,'volumetric')
		gNb = constructW_vol(maskStru,nb);
	elseif strcmp(dataType,'surface')
		gNb = constructW_surf(maskStru,nb,auxMask);
	end
else
	error('Data format not supported !');
end

