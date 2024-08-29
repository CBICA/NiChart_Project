function [U, V, lambdas, iterLog] = mMultiNMF_l21p1_ard(X, W, D, L, Wt, Dt, Lt, initU, initV, options, outDir)
% with ARD regularization on U and V

viewNum = length(X);

Rounds = options.rounds;
maxInIter = options.maxIter;
minInIter = options.minIter;

load([outDir '/JobStatus.mat']);
if strcmp(JobStatus, 'Preprocessing') | strcmp(JobStatus, 'Iteration_0')
    oldL = Inf;
    j = 0;
    iterLog = [];
    JobStatus = 'Iteration_0';
    save([outDir '/JobStatus.mat'], 'JobStatus');
    
    U = initU;
    V = initV;
    clear initU;
    clear initV;
else
    load([outDir '/Iteration_res.mat']);
    load([outDir '/Iteration_error.mat']);
    load([outDir '/JobStatus.mat']);
    j = str2num(JobStatus(11:end));
    oldL = iterLog(end - 2);
end

% 
if isfield(options,'ardUsed') && options.ardUsed>0
    disp('mMultiNMF_l21p1 with ard...');
    hyperLam = zeros(viewNum,1);
    lambdas = cell(viewNum,1);
    eta = options.eta;
    for vi=1:viewNum
        [mFea,nSmp] = size(X{vi});
        lambdas{vi} = sum(U{vi}) / mFea;
        
        hyperLam(vi) = eta * sum(sum(X{vi}.^2)) / (mFea*nSmp*2);
    end
else
    disp('mMultiNMF_l21p1...');
end

while j < Rounds
    % calculate current objective function value
    j = j + 1;
    
    tmpl21 = zeros(size(V{1}));
    L1 = 0;
    ardU = 0;
    tmp1 = 0;
    tmp2 = 0;
    tmp3 = 0;
    
    for i = 1:viewNum
        [mFea,nSmp] = size(X{i});
        
        tmpl21 = tmpl21 + V{i}.^2;
        
        if isfield(options,'alphaS1')
            tmpNorm1 = sum(V{i},1);
            tmpNorm2 = sqrt(sum(V{i}.^2,1));
            L1 = L1 + options.alphaS1 * sum(tmpNorm1./max(tmpNorm2,eps));
        end
        
        % ard term for U
        if isfield(options,'ardUsed') && options.ardUsed>0
            su = sum(U{i});
            su(su==0) = 1;
            ardU = ardU + sum(log(su))*mFea*hyperLam(i);
        end
        
        tmpDf = (X{i}-U{i}*V{i}').^2;
        tmp1 = tmp1 + sum(tmpDf(:));
        
        if isfield(options,'alphaL')
            dVi = double(V{i}');
            tmp2 = tmp2 + dVi * L{i} .* dVi;
        end
        
        if isfield(options,'alphaLT')
            dUi = double(U{i}');
            tmp3 = tmp3 + dUi * Lt{i} .* dUi;
        end
    end
    L21 = options.alphaS21 * sum(sum(sqrt(tmpl21))./max(sqrt(sum(tmpl21)),eps));
    Ldf = tmp1;
    Lsl = sum(tmp2(:));
    Ltl = sum(tmp3(:));
    
    logL = L21 + ardU + Ldf + Lsl + Ltl + L1;
    
    iterLog(end+1) = logL;
    disp(['  round:',num2str(j),' logL:',num2str(logL),',dataFit:',num2str(Ldf)...
        ',spaLap:',num2str(Lsl),',temLap:',num2str(Ltl),',L21:',num2str(L21),...
        ',L1:',num2str(L1),',ardU:',num2str(ardU)]);
    
    save([outDir '/Iteration_error.mat'], 'oldL', 'iterLog');
    if j>5 && (oldL-logL)/max(oldL,eps)<options.error
        break;
    end
    
    oldU = U;
    oldV = V;
    oldL = logL;
    
    for i=1:viewNum
        disp(i);
        [mFea,nSmp] = size(X{i});
        
        iter = 0;
        oldInLogL = inf;
        
        fixl2 = zeros(size(V{1}));
        for vi = 1:viewNum
            if vi~=i
                fixl2 = fixl2 + V{vi}.^2;
            end
        end
        
        while iter<maxInIter
            iter = iter + 1;
            
            % ===================== update V ========================
            XU = X{i}'*U{i};
            UU = U{i}'*U{i};
            VUU = V{i}*UU;
            
            tmpl2 = fixl2 + V{i}.^2;
            if options.alphaS21>0
                tmpl21 = sqrt(tmpl2);
                tmpl22 = repmat(sqrt(sum(tmpl2,1)),nSmp,1);
                tmpl21s = repmat(sum(tmpl21,1),nSmp,1);
                posTerm = V{i} ./ max(tmpl21.*tmpl22,eps);
                negTerm = V{i} .* tmpl21s ./ max(tmpl22.^3,eps);
                
                VUU = VUU + 0.5 * options.alphaS21 * posTerm;
                XU = XU + 0.5 * options.alphaS21 * negTerm;
            end
            
            if isfield(options,'alphaL')
                WV = W{i} * double(V{i});
                DV = D{i} * double(V{i});
                
                XU = XU + WV;
                VUU = VUU + DV;
            end
            
            if isfield(options,'alphaS1')
                sV = max(repmat(sum(V{i}),nSmp,1),eps);
                normV = sqrt(sum(V{i}.^2));
                normVmat = repmat(normV,nSmp,1);
                posTerm = 1./max(normVmat,eps);
                negTerm = V{i}.*sV./max(normVmat.^3,eps);
                
                XU = XU + 0.5*options.alphaS1*negTerm;
                VUU = VUU + 0.5*options.alphaS1*posTerm;
            end
            
            V{i} = V{i}.*(XU./max(VUU,eps));
            
            prunInd = sum(V{i}~=0)==1;
            if any(prunInd)
                V{i}(:,prunInd) = zeros(nSmp,sum(prunInd));
                U{i}(:,prunInd) = zeros(mFea,sum(prunInd));
            end
            
            % ==== normalize U and V ====
            [U{i},V{i}] = Normalize(U{i}, V{i});
            
            % ===================== update U =========================
            XV = X{i}*V{i};
            VV = V{i}'*V{i};
            UVV = U{i}*VV;
            
            if isfield(options,'ardUsed') && options.ardUsed>0 % ard term for U
                posTerm = 1./max(repmat(lambdas{i},mFea,1),eps);
                UVV = UVV + posTerm*hyperLam(i);
            end
            
            if isfield(options,'alphaLT')
                WU = Wt{i} * double(U{i});
                DU = Dt{i} * double(U{i});
                
                XV = XV + WU;
                UVV = UVV + DU;
            end
            
            U{i} = U{i}.*(XV./max(UVV,eps));
            %U{i}(U{i}<1e-6) = 0;
            
            prunInd = sum(U{i})==0;
            if any(prunInd)
                V{i}(:,prunInd) = zeros(nSmp,sum(prunInd));
                U{i}(:,prunInd) = zeros(mFea,sum(prunInd));
            end
            
            % update lambda
            if isfield(options,'ardUsed') && options.ardUsed>0
                lambdas{i} = sum(U{i}) / mFea;
            end
            % ==== calculate partial objective function value ====
            inTl = 0;
            inSl = 0;
            LardU = 0;
            LL1 = 0;
            
            inDf = (X{i}-U{i}*V{i}').^2;
            
            if isfield(options,'alphaLT')
                dUi = double(U{i}');
                inTl = dUi * Lt{i} .* dUi;
            end
          
            if isfield(options,'ardUsed') && options.ardUsed>0
                % ard term for U
                su = sum(U{i});
                su(su==0) = 1;
                LardU = sum(log(su))*mFea*hyperLam(i);
            end
            inL21 = zeros(size(V{1}));
            if options.alphaS21>0
                for vi=1:viewNum
                    inL21 = inL21 + V{vi}.^2;
                end
            end
            if isfield(options,'alphaS1')
                tmpNorm1 = sum(V{i},1);
                tmpNorm2 = sqrt(sum(V{i}.^2,1));
                LL1 = options.alphaS1 * sum(tmpNorm1./max(tmpNorm2,eps));
            end
            
            inL21 = sum(sqrt(inL21))./max(sqrt(sum(inL21)),eps);
            LDf = sum(inDf(:));
            LTl = sum(inTl(:));
            LSl = sum(inSl(:));
            LL21 = options.alphaS21 * sum(inL21(:));
            
            inLogL = LDf + LTl + LSl + LardU + LL21 + LL1;
            
            if iter>minInIter && abs(oldInLogL-inLogL)/max(oldInLogL,eps)<options.error
                break;
            end
            oldInLogL = inLogL;
        end
    end
    save([outDir '/Iteration_res.mat'], 'W', 'D', 'L', 'Wt', 'Dt', 'Lt', 'U', 'V', '-v7.3');
    JobStatus = ['Iteration_' num2str(j)];
    save([outDir '/JobStatus.mat'], 'JobStatus');
end

end % function


function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 1, 1);
end


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
            V = bsxfun(@times, V, norms);
        end
    end
end


