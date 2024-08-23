function gsurf = constructW_surf(surfStru, nb, surfMask)
% surf: a structure containing the surface topology, includes
%       surfStru.vx_l: vertices of left hemi-sphere
%       surfStru.vx_r: vertices of right hemi-sphere
%       surfStru.faces_l: faces of left hemi-sphere
%       surfStru.faces_r: faces of right hemi-sphere
% nb: size of spaital neighborhood
% surfMask: a structure containing the mask of regions of interest (ROI) for each hemi-sphere
%           surfMask.l (vector) and surfMask.r (vector)

if nargin==1
    nb = 1;
    surfMask = [];
elseif nargin==2
    surfMask = [];
end

vxNum_l = length(surfStru.vx_l);
vxNum_r = length(surfStru.vx_r);

if isempty(surfMask)
    surfMask.l = ones(vxNum_l,1);
    surfMask.r = ones(vxNum_r,1);
end

vxNum_l_av = sum(surfMask.l>0);
vxNum_r_av = sum(surfMask.r>0);

%
mapper_l = zeros(size(surfMask.l));
mapper_r = zeros(size(surfMask.r));

mapper_l(surfMask.l>0) = 1:vxNum_l_av;
mapper_r(surfMask.r>0) = 1:vxNum_r_av;

%
vxNum = vxNum_l_av + vxNum_r_av;
gsurf = cell(vxNum,1);

if nb==1
    for fi=1:length(surfStru.faces_l)
        curF = surfStru.faces_l(fi,:);
        for sti=1:length(curF)
            mCurF_s = mapper_l(curF(sti));
            if mCurF_s>0
                for eni=sti+1:length(curF)
                    mCurF_e = mapper_l(curF(eni));
                    if mCurF_e>0
                        gsurf{mCurF_s}(end+1) = mCurF_e;
                        gsurf{mCurF_e}(end+1) = mCurF_s;
                    end
                end
            end
        end
    end

    for fi=1:length(surfStru.faces_r)
        curF = surfStru.faces_r(fi,:);
        for sti=1:length(curF)
            mCurF_s = mapper_r(curF(sti));
            if mCurF_s>0
                for eni=sti+1:length(curF)
                    mCurF_e = mapper_r(curF(eni));
                    if mCurF_e>0
                        gsurf{mCurF_s+vxNum_l_av}(end+1) = mCurF_e + vxNum_l_av;
                        gsurf{mCurF_e+vxNum_l_av}(end+1) = mCurF_s + vxNum_l_av;
                    end
                end
            end
        end
    end
else
    gl = sparse(vxNum_l_av,vxNum_l_av);
    for fi=1:length(surfStru.faces_l)
        curF = surfStru.faces_l(fi,:);
        for sti=1:length(curF)
            mCurF_s = mapper_l(curF(sti));
            if mCurF_s>0
                for eni=sti+1:length(curF)
                    mCurF_e = mapper_l(curF(eni));
                    if mCurF_e>0
                        gl(mCurF_s,mCurF_e) = 1;
                        gl(mCurF_e,mCurF_s) = 1;
                    end
                end
            end
        end
    end

    for vi=1:vxNum_l_av
        order = graphtraverse(gl,vi,'Depth',nb,'Directed','false');
        gsurf{vi} = order;
    end

    gr = sparse(vxNum_r_av,vxNum_r_av);
    for fi=1:length(surfStru.faces_r)
        curF = surfStru.faces_r(fi,:);
        for sti=1:length(curF)
            mCurF_s = mapper_r(curF(sti));
            if mCurF_s>0
                for eni=sti+1:length(curF)
                    mCurF_e = mapper_r(curF(eni));
                    if mCurF_e>0
                        gr(mCurF_s,mCurF_e) = 1;
                        gr(mCurF_e,mCurF_s) = 1;
                    end
                end
            end
        end
    end

    for vi=1:vxNum_r_av
        order = graphtraverse(gr,vi,'Depth',nb,'Directed','false');
        gsurf{vi+vxNum_l_av} = order + vxNum_l_av;
    end
end


