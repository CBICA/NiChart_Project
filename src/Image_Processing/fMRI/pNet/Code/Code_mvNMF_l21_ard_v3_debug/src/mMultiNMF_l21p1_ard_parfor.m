function [U, V, lambdas, iterLog] = mMultiNMF_l21p1_ard_parfor(X, W, D, L, Wt, Dt, Lt, initU, initV, options)
% with ARD regularization on U and V

viewNum = length(X);

Rounds = options.rounds;
maxInIter = options.maxIter;
minInIter = options.minIter;

U = initU;
V = initV;

clear initU;
clear initV;

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

% matlabpool open using 5 workers
 %matlabpool('open',5);
% disp(['matlabpool size: ',num2str(parpool('size'))]);

parpool(5);
%



oldL = Inf;
j = 0;
iterLog = 0;
restartJ = 0; % added by hmli
while j < Rounds
	% calculate current objective function value
    j = j + 1;
	
    tmpl21 = zeros(size(V{1}));
    L1 = 0;
    ardU = 0;
    tmp1 = 0;
    tmp2 = 0;
    tmp3 = 0;
    
    % parfor
    if isfield(options,'alphaS1')
        parAlphaS1 = options.alphaS1;
    else
        parAlphaS1 = 0;
    end
    if isfield(options,'ardUsed')
        parArdUsed = options.ardUsed;
    else
        parArdUsed = 0;
    end

    if isfield(options,'alphaL')
        parAlphaL = options.alphaL;
    else
        parAlphaL = 0;
    end
    if isfield(options,'alphaLT')
        parAlphaLT = options.alphaLT;
    else
        parAlphaLT = 0;
    end
    
    parfor i = 1:viewNum
        [mFea,nSmp] = size(X{i});

		tmpl21 = tmpl21 + V{i}.^2;
        
        if parAlphaS1>0
            tmpNorm1 = sum(V{i},1);
            tmpNorm2 = sqrt(sum(V{i}.^2,1));
            L1 = L1 + parAlphaS1 * sum(tmpNorm1./max(tmpNorm2,eps));
        end

        % ard term for U
        if parArdUsed>0
            su = sum(U{i});
            su(su==0) = 1;
            ardU = ardU + sum(log(su))*mFea*hyperLam(i);
        end
        
        tmpDf = (X{i}-U{i}*V{i}').^2;
        tmp1 = tmp1 + sum(tmpDf(:));
		
        if parAlphaL>0
            dVi = double(V{i}');
            tmp2 = tmp2 + dVi * L{i} .* dVi;
        end
		
        if parAlphaLT>0
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
    %disp(['  round:',num2str(j),' logL:',num2str(logL)]);
    disp(['  round:',num2str(j),' logL:',num2str(logL),',dataFit:',num2str(Ldf)...
          ',spaLap:',num2str(Lsl),',temLap:',num2str(Ltl),',L21:',num2str(L21),...
          ',L1:',num2str(L1),',ardU:',num2str(ardU)]);
    
%    if (oldL < logL) 
%        % modified by hmli
%         if restartJ == 0
%             restartJ = j;
%             consRestart = 1;
%         elseif restartJ == j
%             consRestart = consRestart + 1;
%             if consRestart>=3
% 				U = oldU;
% 				V = oldV;
% 				logL = oldL;	
% 			
%                 break;
%             end
%         else
%             restartJ = j;
%             consRestart = 1;
%         end
%         
%         j = j - 1;
%         %disp('restrart this iteration');
%    else
        if j>5 && (oldL-logL)/max(oldL,eps)<options.error
            break;
        end
%    end
    
    %oldU = U;
    %oldV = V;
    oldL = logL;
	
    % parfor
    if isfield(options,'alphaS21')
        parAlphaS21 = options.alphaS21;
    else
        parAlphaS21 = 0;
    end
    if isfield(options,'alphaS1')
        parAlphaS1 = options.alphaS1;
    else
        parAlphaS1 = 0;
    end
    if isfield(options,'ardUsed')
        parArdUsed = options.ardUsed;
    else
        parArdUsed = 0;
    end	
    if isfield(options,'alphaL')
        parAlphaL = options.alphaL;
    else
        parAlphaL = 0;
    end
    if isfield(options,'alphaLT')
        parAlphaLT = options.alphaLT;
    else
        parAlphaLT = 0;
    end
    parError = options.error;
    
    fixl2_ = zeros(size(V{1}));
    for vi = 1:viewNum
        fixl2_ = fixl2_ + V{vi}.^2;
    end
	parfor i=1:viewNum
        [mFea,nSmp] = size(X{i});

		iter = 0;
		oldInLogL = inf;
		
        fixl2 = fixl2_ - V{i}.^2;
        
        breakFlag = 0;
        while (iter<maxInIter && breakFlag==0)
			iter = iter + 1;
            
			% ===================== update V ========================
			XU = X{i}'*U{i};  
			UU = U{i}'*U{i};  
			VUU = V{i}*UU;
			
			tmpl2 = fixl2 + V{i}.^2;
            if parAlphaS21>0
                tmpl21 = sqrt(tmpl2);
                tmpl22 = repmat(sqrt(sum(tmpl2,1)),nSmp,1);
                tmpl21s = repmat(sum(tmpl21,1),nSmp,1);
                posTerm = V{i} ./ max(tmpl21.*tmpl22,eps);
                negTerm = V{i} .* tmpl21s ./ max(tmpl22.^3,eps);

                VUU = VUU + 0.5 * parAlphaS21 * posTerm;
                XU = XU + 0.5 * parAlphaS21 * negTerm;
            end
            
            if parAlphaL>0
                WV = W{i} * double(V{i});
                DV = D{i} * double(V{i});

                XU = XU + WV; 
                VUU = VUU + DV;
            end
			
            if parAlphaS1>0
                sV = max(repmat(sum(V{i}),nSmp,1),eps);
                normV = sqrt(sum(V{i}.^2));
                normVmat = repmat(normV,nSmp,1);
                posTerm = 1./max(normVmat,eps);
                negTerm = V{i}.*sV./max(normVmat.^3,eps);
                
                XU = XU + 0.5*parAlphaS1*negTerm;
                VUU = VUU + 0.5*parAlphaS1*posTerm;
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
            
            if parArdUsed>0  % ard term for U
                posTerm = 1./max(repmat(lambdas{i},mFea,1),eps);
                UVV = UVV + posTerm*hyperLam(i);
            end
			
			if parAlphaLT>0
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
            if parArdUsed>0
                lambdas{i} = sum(U{i}) / mFea;
            end
			% ==== calculate partial objective function value ====
            inTl = 0;
            inSl = 0;
            LardU = 0;
            LL1 = 0;
            
            inDf = (X{i}-U{i}*V{i}').^2;

            if parAlphaLT>0
				dUi = double(U{i}');
				inTl = dUi * Lt{i} .* dUi;
            end
            if parAlphaL>0
				dVi = double(V{i}');
				inSl = dVi * L{i} .* dVi;
            end
            if parArdUsed>0
                % ard term for U
                su = sum(U{i});
                su(su==0) = 1;
                LardU = sum(log(su))*mFea*hyperLam(i);
            end
            
            inL21 = fixl2 + V{i}.^2;
			
            if parAlphaS1>0
                tmpNorm1 = sum(V{i},1);
                tmpNorm2 = sqrt(sum(V{i}.^2,1));
                LL1 = parAlphaS1 * sum(tmpNorm1./max(tmpNorm2,eps));
            end
            
            inL21 = sum(sqrt(inL21))./max(sqrt(sum(inL21)),eps);
			LDf = sum(inDf(:));
			LTl = sum(inTl(:));
			LSl = sum(inSl(:));
			LL21 = parAlphaS21 * sum(inL21(:));
            
			inLogL = LDf + LTl + LSl + LardU + LL21 + LL1;
            
            breakFlag = iter>minInIter && abs(oldInLogL-inLogL)/max(oldInLogL,eps)<parError;
			
			oldInLogL = inLogL;
        end      
	end
end

 %matlabpool close

delete(gcp('nocreate'))
%

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


