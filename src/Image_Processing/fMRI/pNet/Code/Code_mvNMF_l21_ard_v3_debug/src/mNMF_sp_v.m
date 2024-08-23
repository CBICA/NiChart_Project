function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = mNMF_sp_v(X, k, W, options, U_, V_)
% Non-negative Matrix Factorization (NMF) with multiplicative update
% with sparse and graph constraints
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of time points
%       nSmp  ... number of voxels
% k ... number of ICN
% W ... weight matrix of the sample affinity graph
% options ... Structure holding all settings
%
% U_ ... initialization for time series 
% V_ ... initialization for spatial maps
%
%   Written by Deng Cai (dengcai AT gmail.com)
%   Modified by Jialu Liu (jliu64 AT illinois.edu)
%   Modified by hmli

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;

alphaL = options.L;
alphaS = options.S1;

Norm = 1;
NormV = 1;

[mFea,nSmp] = size(X);

if alphaL > 0
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
    
    D_mhalf = spdiags(DCol.^-0.5,0,nSmp,nSmp) ;
    L = D_mhalf*L*D_mhalf * alphaL;
    W = D_mhalf*W*D_mhalf * alphaL;
    D = D_mhalf*D*D_mhalf * alphaL;
else
    L = [];
end

bSuccess.bSuccess = 1;

selectInit = 1;
mean_X = sum(X(:))/(mFea*nSmp);
if isempty(U_)
    U = (rand(mFea,k)+1)*(sqrt(mean_X/k));
    if isempty(V_)
        V = (rand(nSmp,k)+1)*(sqrt(mean_X/k));
    else
        V = V_;
    end
else
    U = U_;
    if isempty(V_)
        V = (rand(nSmp,k)+1)*(sqrt(mean_X/k));
    else
        V = V_;
    end
end

[U,V] = NormalizeUV(U, V, NormV, Norm);

if isfield(options,'ard') && options.ard==1
    ard = 1;
	eta = 1;
	lambdas = sum(U) / mFea;
	hyperLam = eta * sum(sum(X.^2)) / (mFea*nSmp*2);
else
    ard = 0;
    hyperLam = 0;
end
	
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    %minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, L, alphaS, ard, hyperLam);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, L, alphaS, ard, hyperLam);
            meanFit = objhistory*10;
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
        XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU = V*UU; % nk^2
        
        if alphaS>0
%             % L1 sparsity
%             F = ones(size(V));
%             VUU = VUU + 0.5*alphaS * F;
            
            % scale-invariant sparsity
            tmpNorm2 = sqrt(sum(V.^2,1));
            posTerm = 1 ./ max(repmat(tmpNorm2,nSmp,1),eps);
            tmpNorm1 = sum(V,1);
            negTerm = V .* repmat(tmpNorm1,nSmp,1) ./ max(repmat(tmpNorm2.^3,nSmp,1),eps);
            
            XU = XU + 0.5*alphaS * negTerm;
            VUU = VUU + 0.5*alphaS * posTerm;
        end
        if alphaL>0
            V = double(V);
            WV = W*V;
            DV = D*V;
            
            XU = XU + WV;
            VUU = VUU + DV;
        end
        
        V = V.*(XU./max(VUU,eps));
        
        prunInd = sum(V~=0)==1;
        if any(prunInd)
           V(:,prunInd) = zeros(nSmp,sum(prunInd));
           U(:,prunInd) = zeros(mFea,sum(prunInd));
        end
        
        % erease very small numbers
        %ereEle = V./max(eps,repmat(max(V),size(V,1),1));
        %V(ereEle<1e-6) = 0; 
        
        [U, V] = NormalizeUV(U, V, NormV, Norm);
        
		% ===================== update U ========================
%        XV = X*V;   % mnk or pk (p<<mn)
%        VV = V'*V;  % nk^2
%        UVV = U*VV; % mk^2

        % needed if Ui is restricted to norm 1
%         if alphaS>0
%             % L1 sparsity
%             F = repmat(sum(V), mFea, 1);
%             UVV = UVV + 0.5*alphaS * F;
%         end
%         if alphaL>0
%             V = double(V);
%             VLV = repmat(diag(V'*L*V)' .* sum(U,1), mFea, 1);
%             UVV = UVV + VLV;
%         end
		
%		if isfield(options,'ard') && options.ard==1
%			posTerm = 1./max(repmat(lambdas,mFea,1),eps);
%            UVV = UVV + posTerm*hyperLam;
%		end
            
%        U = U.*(XV./max(UVV,eps));
        
%        prunInd = sum(U)==0;
%        if any(prunInd)
%           V(:,prunInd) = zeros(nSmp,sum(prunInd));
%           U(:,prunInd) = zeros(mFea,sum(prunInd));
%        end
        
%        if isfield(options,'ard') && options.ard==1
%            lambdas = sum(U) / mFea;
%        end
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, L, alphaS, ard, hyperLam);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, L, alphaS, ard, hyperLam);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = abs((meanFit-newobj)/meanFit);
                else
                    if isfield(options,'Converge') && options.Converge
                        [newobj,newObjStr] = CalculateObj(X, U, V, L, alphaS, ard, hyperLam);
                        objhistory = [objhistory newobj];
                        if mod(nIter,10)==0
                           disp(['  iter: ',num2str(nIter)]);
                           disp(newObjStr);
                        end
                        
                        meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                        maxErr = abs((meanFit-newobj)/meanFit);
                        %maxErr = abs(objhistory(end)-objhistory(end-1))/abs(objhistory(end));
                    else
                        maxErr = 1;
                    end
                    
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
			    U = (rand(mFea,k)+1)*(sqrt(mean_X/k));
			    if isempty(V_)
			        V = (rand(nSmp,k)+1)*(sqrt(mean_X/k));
			    else
			        V = V_;
			    end
			else
			    U = U_;
			    if isempty(V_)
			        V = (rand(nSmp,k)+1)*(sqrt(mean_X/k));
			    else
			        V = V_;
			    end
			end

            [U,V] = NormalizeUV(U, V, NormV, Norm);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
    
end

nIter_final = nIter_final + minIterOrig;

[U_final, V_final] = NormalizeUV(U_final, V_final, NormV, Norm);
objhistory_final = CalculateObj(X, U_final, V_final, L, alphaS, ard, hyperLam); % added by hmli

%==========================================================================

function [obj, objStr] = CalculateObj(X, U, V, L, alphaS, ard, hyperLam, deltaVU, dVordU)
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
    if isempty(L)
        obj_Lap = 0;
    else
        V = double(V);
        obj_Lap = sum(sum((L*V).*V));
    end
    
    %obj_Spa = alphaS * sum(sum(V));
    tmpNorm1 = sum(V,1);
    tmpNorm2 = sqrt(sum(V.^2,1)) + eps;
    obj_Spa = alphaS * sum(tmpNorm1./tmpNorm2);
    
    if ard>0
        su = sum(U);
        su(su==0) = 1;
        obj_ard = sum(log(su))*mFea*hyperLam;
    else
        obj_ard = 0;
    end
    
    obj = obj_NMF + obj_Lap + obj_Spa;
    objStr = ['    totObj:',num2str(obj),',NMF:',num2str(obj_NMF),',Lap:',num2str(obj_Lap),',Spa:',num2str(obj_Spa),',Ard:',num2str(obj_ard)];
    

% function [U, V] = Normalize(U, V)
%     [U,V] = NormalizeUV(U, V, 1, 1);


function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,eps);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,eps);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            %norms = sum(abs(V),1);
            norms = max(V);
            norms = max(norms,eps);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            %norms = sum(abs(U),1);
            norms = max(U);
            norms = max(norms,eps);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    end


