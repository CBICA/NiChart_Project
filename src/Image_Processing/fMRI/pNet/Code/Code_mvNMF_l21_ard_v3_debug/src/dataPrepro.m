function pX = dataPrepro(X,alg,norm)
% preprocess the data X
% X: m x n matrix, m - number of variables, n - number of observations
    switch lower(alg)
        case 'z'
            % standard score for each variable
            mVec = mean(X,2);
            sVec = max(std(X,0,2),eps);
            pX = (X-repmat(mVec,1,size(X,2)))./repmat(sVec,1,size(X,2));
        case 'gp'
            % remove negative value globally
            minVal = min(X(:));
            shiftVal = abs(min(minVal,0));
            pX = X + shiftVal;
        case 'vp'
            % remove negative value voxel-wisely
            minVal = min(X,[],1);
            shiftVal = abs(min(minVal,0));
            pX = X + repmat(shiftVal,size(X,1),1);
        otherwise
            % do nothing
            disp('  unknown preprocess parameters, no preprocess applied');
            pX = X;
    end
        
    % normalization
    switch lower(norm)
        case 'n2'
            % l2 normalization for each observation
            l2norm = sqrt(sum(pX.^2)) + eps;
            pX = pX ./ repmat(l2norm,size(pX,1),1);
        case 'n1'
            % l1 normalization for each observation
            l1norm = sum(pX) + eps;
            pX = pX ./ repmat(l1norm,size(pX,1),1);
        case 'rn1'
            % l1 normalization for each variable
            l1norm = sum(pX,2) + eps;
            pX = pX ./ repmat(l1norm,1,size(pX,2));
        case 'g'
            % global scale        
            [sVal,sInd] = sort(pX(:));
            perT = 0.001;
            minVal = sVal(round(length(sInd)*perT));
            maxVal = sVal(round(length(sInd)*(1-perT)));
            pX(pX<minVal) = minVal;
            pX(pX>maxVal) = maxVal;
            pX = (pX-minVal)/max((maxVal-minVal),eps);
        case 'vmax'
            cmin = repmat(min(pX),size(pX,1),1);
            cmax = repmat(max(pX),size(pX,1),1);
            pX = (pX-cmin)./max(eps,cmax-cmin);
        otherwise
            % do nothing
            disp('  unknown normalization parameters, no normalization applied');
    end
    if any(isnan(pX))
        error('  nan exists, check the preprocessed data');
    end
end


