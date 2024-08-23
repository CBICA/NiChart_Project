function [U_final, V_final] = backNMF_v(X, k, options, U_, V_)
% Non-negative Matrix Factorization (NMF) with multiplicative update
%
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
%
% options ... Structure holding all settings
%
% U_ ... initialization for basis matrix 
% V_ ... initialization for coefficient matrix 
%
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

[mFea,nSmp] = size(X);

bSuccess.bSuccess = 1;

selectInit = 1;
if isempty(U_)
    U = abs(rand(mFea,k));
    norms = sqrt(sum(U.^2,1));
    norms = max(norms,1e-10);
    U = U./repmat(norms,mFea,1);
    if isempty(V_)
        V = abs(rand(nSmp,k));
        V = V/sum(sum(V));
    else
        V = V_;
    end
else
    U = U_;
    if isempty(V_)
        V = abs(rand(nSmp,k));
        V = V/sum(sum(V));
    else
        V = V_;
    end
end

%[U,V] = NormalizeUV(U, V, NormV, Norm);
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

% added by hmli
%disp(num2str(CalculateObj(X, U, V)));

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
        
        V = V.*(XU./max(VUU,1e-10));
        
        % ===================== update U ========================
%         XV = X*V;   % mnk or pk (p<<mn)
%         VV = V'*V;  % nk^2
%         UVV = U*VV; % mk^2
%         
%         U = U.*(XV./max(UVV,1e-10)); % 3mk
        % added by hmli
        %newobj = CalculateObj(X, U, V);
        %disp(num2str(newobj));
        
%        [U,V] = Normalize(U, V);
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V);
                        objhistory = [objhistory newobj]; %#ok<AGROW>
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
                norms = max(norms,1e-10);
                U = U./repmat(norms,mFea,1);
                if isempty(V_)
                    V = abs(rand(nSmp,k));
                    V = V/sum(sum(V));
                else
                    V = V_;
                end
            else
                U = U_;
                if isempty(V_)
                    V = abs(rand(nSmp,k));
                    V = V/sum(sum(V));
                else
                    V = V_;
                end
            end

            %[U,V] = NormalizeUV(U, V, NormV, Norm);
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
    
    %display(['  tryNo: ',num2str(tryNo),' ','obj: ',num2str(objhistory(end))]);% added by hmli
end

%nIter_final = nIter_final + minIterOrig;

%[U_final, V_final] = Normalize(U_final, V_final);


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
   %obj_Lap = alpha*sum(sum((L*V).*V));
   
    obj = obj_NMF;


function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 1, 1);


function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = max(abs(V));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            %norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    end

        