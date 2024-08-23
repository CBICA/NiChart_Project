function gvol = constructW_vol(maskMat,neiR)
%generate sparse graph for 3D image
% maskMat: a 3D matrix containing the mask of regions of intesest (ROI)
% neiR: the radius of spatial neighborhood

if nargin<2
    neiR = 1;
end

maskMat = int32(maskMat);
maskSz = size(maskMat);
if length(maskSz)~=3
    error('  Only support 3D image now');
end

if length(maskSz)~=length(neiR)
    neiRvec = repmat(neiR,1,length(maskSz));
else
    neiRvec = neiR;
end

%assign a label (1 to length(non-zero voxel)) for each non-zero voxel in the mask
labMaskMat = maskMat;
nonZero = maskMat~=0;
eleNum = sum(nonZero(:));
labMaskMat(maskMat~=0) = 1:eleNum;

gvol = cell(eleNum,1);
for zi=1:maskSz(3)
    for yi=1:maskSz(2)
        for xi=1:maskSz(1)
            cenLab = labMaskMat(xi,yi,zi);
            if cenLab~=0
                nstart = max([xi,yi,zi]-neiRvec,[1,1,1]);
                nend = min([xi,yi,zi]+neiRvec,maskSz);
				for zni=nstart(3):nend(3)
					for yni=nstart(2):nend(2)
						for xni=nstart(1):nend(1)
							neiLab = labMaskMat(xni,yni,zni);
							if neiLab~=0 && neiLab>cenLab
								gvol{cenLab}(end+1) = neiLab;
                                gvol{neiLab}(end+1) = cenLab;
							end
						end
					end
				end
            end
        end
    end
end

            
