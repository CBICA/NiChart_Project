function [U_final, V_final] = backNMF_u(X, k, options, U_, V_)
% Nonnegative regression for U with multiplicative update
%
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of features (time points for fMRI)
%       nSmp  ... number of samples (number of voxels)
% k ... number of hidden factors
%
% options ... Structure holding all settings
%
% U_ ... initialization for U (time series)
% V_ ... initialization for V (spatial maps), V_ should not be empty
%
% Modified by hmli

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;

[mFea,nSmp] = size(X);

bSuccess.bSuccess = 1;

selectInit = 1;
if isempty(U_)
    U = abs(rand(mFea,k));
    norms = sqrt(sum(U.^2,1));
    norms = max(norms,eps);
    U = U./repmat(norms,mFea,1);
else
    U = U_;
end
V = V_;

if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

tryNo = 0;
while tryNo < nRepeat   
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    nStepTrial = 0;
    while(maxErr > differror)
        % ===================== update V ========================
%        XU = X'*U;  % mnk or pk (p<<mn)
%        UU = U'*U;  % mk^2
%        VUU = V*UU; % nk^2
        
%        V = V.*(XU./max(VUU,eps));
        
        % ===================== update U ========================
        XV = X * V;   % mnk or pk (p<<mn)
        VV = V' * V;  % nk^2
        UVV = U * VV; % mk^2
        
        U = U .* (XV./max(UVV,eps)); % 3mk
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V);
                        objhistory = [objhistory newobj];
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
                
            end
        end
    end
    
    elapse = cputime - tmp_T;

    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
        bSuccess.nStepTrial = nStepTrial;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
           bSuccess.nStepTrial = nStepTrial;
           if selectInit
               elapse_final = elapse;
           else
               elapse_final = elapse_final+elapse;
           end
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            if isempty(U_)
                U = abs(rand(mFea,k));
                norms = sqrt(sum(U.^2,1));
                norms = max(norms,eps);
                U = U./repmat(norms,mFea,1);
            else
                U = U_;
            end
            V = V_;
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            %V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, deltaVU, dVordU)
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    maxM = 62500000;
    [mFea, nSmp] = size(X);
    mn = numel(X);
    nBlock = floor(mn*3/maxM);

    if mn < maxM
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV ;
            end
        end
    end
   
    obj = obj_NMF;


