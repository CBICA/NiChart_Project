function deployFuncMvnmfL21p1_func_surf_hcp_single_gV_17(sbjListFile,sbjTCFolder,wbPath,prepDataFile,outDir,resId,initName,K,alphaS21,alphaL,vxI,spaR,ard,eta,iterNum,calcGrp,parforOn)
    % Yuncong Ma, 1/9/2023
    % Modified from deployFuncMvnmfL21p1_func_surf_hcp_single
    % Following Zheng Zhou's modification
    % deployFuncMvnmfL21p1_func_vol_single.m
    % Input only includes one data file
    % Use the group-level V for regularization and sparsity
    % File_GroupV contains initV in .mat format
    % This function add modified func_mMvNMF4fmri_l21p1_ard_woSrcLoad and mMultiNMF_l21p1_ard
    % No more support for parforOn
    
    if nargin~=17
        error('number of input should be 17 !');
    end
    
    if isdeployed || ischar(K)
        K = str2double(K);
        alphaS21 = str2double(alphaS21);
        alphaL = str2double(alphaL);
        vxI = str2double(vxI);
        spaR = str2double(spaR);
        ard = str2double(ard);
        eta = str2double(eta);
        iterNum = str2double(iterNum);
        calcGrp = str2double(calcGrp);
    end
    
    Debug=1;
    if Debug==1
        fprintf('\n--deployFuncMvnmfL21p1_func_surf_hcp_single_gV_17--\n');
        display(sbjListFile);
        display(sbjTCFolder);
        display(wbPath);
        display(prepDataFile);
        display(outDir);
        display(resId);
        display(initName);
        display(K);
        display(alphaS21);
        display(alphaL);
        display(vxI);
        display(spaR);
        display(ard);
        display(eta);
        display(iterNum);
        display(calcGrp);
    end
    
    if ~exist([sbjTCFolder, '/sbjData.mat'], 'file')
        sbjData = prepareFuncData_hcp_func(sbjListFile,wbPath,'0');
    
        if ~exist(sbjTCFolder,'dir')
            mkdir(sbjTCFolder);
        end
        %     save([sbjTCFolder, '/sbjData.mat'], 'sbjData', '-v7.3');
        JobStatus = 'Preprocessing';
        save([outDir, '/JobStatus.mat'], 'JobStatus');
    else
        JobStatus = 'Preprocessing';
        if ~exist(outDir,'dir')
            mkdir(outDir);
        end
        save([outDir, '/JobStatus.mat'], 'JobStatus');
        load([sbjTCFolder, '/sbjData.mat']);
    end
    
    sbjNum = length(sbjData);
    if Debug==1
        display(sbjNum);
    end
    
    tic;
    func_mMvNMF4fmri_l21p1_ard_woSrcLoad(sbjData,prepDataFile,outDir,resId,initName,sbjNum,K,...
        alphaS21,alphaL,spaR,vxI,ard,eta,iterNum,calcGrp);
    toc;
    
    if isdeployed
        exit;
    else
        disp('Done!');
    end
    
    end
    
    
    
    
    function [U, V] = func_mMvNMF4fmri_l21p1_ard_woSrcLoad(sbjData,prepData,outDir,resId,initName,sbjNum,K,alphaS21,alphaL,spaR,vxI,ard,eta,iterNum,calcGrp)
    
    %disp(['# of input: ',num2str(nargin)]);
    if nargin~=15
        error('15 input parameters are needed');
    end
    
    alphaS1 = alphaS21;
    if alphaS21*sbjNum>=1
        alphaS21 = round(alphaS21*sbjNum);
    else
        alphaS21 = alphaS21*sbjNum;
    end
    
    % parameter setting
    options = [];
    options.maxIter = 100;
    options.error = 1e-4;
    options.nRepeat = 1;
    options.minIter = 30;
    options.meanFitRatio = 0.1;
    options.rounds = iterNum;
    options.NormW = 1;
    options.eta = eta;
    
    options.alphaS21 = alphaS21;
    options.alphaL = alphaL;
    options.vxlInfo = vxI;
    options.spaR = spaR;
    if ard>0
        options.ardUsed = ard;
    end
    
    % output name
    resDir = [outDir,filesep,resId,'_sbj',num2str(sbjNum),'_comp',num2str(K),...
        '_alphaS21_',num2str(alphaS21),'_alphaL',num2str(alphaL),...
        '_vxInfo',num2str(vxI),'_ard',num2str(ard),'_eta',num2str(eta)];
    if ~exist(resDir,'dir')
        mkdir(resDir);
    end
    
    % load data
    numUsed = sbjNum;
    % numUsed = min(numUsed,length(sbjData));
    % sbjData = sbjData(1:numUsed);
    
    % load preparation, containing gNb
    load(prepData);
    
    % load initialization: initV
    load(initName,'initV');
    
    load([outDir '/JobStatus.mat']);
    
    if strcmp(JobStatus, 'Preprocessing')
        %
        vNum = length(gNb);
        W = cell(numUsed,1);
        Wt = cell(numUsed,1);
        D = cell(size(W));
        L = cell(size(W));
        Dt = cell(size(Wt));
        Lt = cell(size(Wt));
        U = cell(numUsed,1);
        V = cell(numUsed,1);
    
        disp('preprocess...');
        for si=1:numUsed
            fprintf('  sbj%d->',si);
            if mod(si,15)==0
                fprintf('\n');
            end
    
            origSbjData = sbjData{si};
            origSbjData = dataPrepro(origSbjData,'vp','vmax');
            sbjData{si} = origSbjData;
    
            nanSbj = isnan(origSbjData);
            if sum(nanSbj(:))>0
                disp([' nan exists: ','sbj',num2str(si)]);
                return;
            end
    
            % construct the spatial affinity graph
            if vxI==0
                if si==1
                    tmpW = sparse(vNum,vNum);
                    for vi=1:vNum
                        for ni=1:length(gNb{vi})
                            nei = gNb{vi}(ni);
                            tmpW(vi,nei) = 1;
                            tmpW(nei,vi) = 1;
                        end
                    end
                    W{si} = tmpW;
                else
                    W{si} = W{1};
                end
            else
                tmpW = sparse(vNum,vNum);
                for vi=1:vNum
                    for ni=1:length(gNb{vi})
                        nei = gNb{vi}(ni);
                        if vi<nei
                            corrVal = (1+corr(origSbjData(:,vi),origSbjData(:,nei)))/2;
                            if isnan(corrVal)
                                corrVal = 0;
                            end
                            tmpW(vi,nei) = corrVal;
                            tmpW(nei,vi) = corrVal;
                        else
                            continue;
                        end
                    end
                end
                W{si} = tmpW;
            end
    
            % temporal affinity matrix
            if isfield(options,'alphaLT')
                t = size(origSbjData,1);
                Wt{si} = zeros(t,t);
                if isfield(options,'timR')
                    tNei = timR;
                else
                    tNei = 1;
                end
                for tni=1:tNei
                    Wt{si} = Wt{si} + diag(ones(t-tni,1),tni) + diag(ones(t-tni,1),-tni);
                end
            else
                Wt{si} = [];
            end
    
            [mFea,nSmp] = size(sbjData{si});
    
            if isfield(options,'alphaL')
                DCol = full(sum(W{si},2));
                D{si} = spdiags(DCol,0,nSmp,nSmp);
                L{si} = D{si} - W{si};
                if isfield(options,'NormW') && options.NormW
                    D_mhalf = spdiags(DCol.^-0.5,0,nSmp,nSmp);
                    L{si} = D_mhalf*L{si}*D_mhalf * options.alphaL;
                    W{si} = D_mhalf*W{si}*D_mhalf * options.alphaL;
                    D{si} = D_mhalf*D{si}*D_mhalf * options.alphaL;
                end
            else
                D{si} = [];
                L{si} = [];
            end
    
            if isfield(options,'alphaLT')
                DCol = full(sum(Wt{si},2));
                Dt{si} = spdiags(DCol,0,mFea,mFea);
                Lt{si} = Dt{si} - Wt{si};
                if isfield(options,'NormW') && options.NormW
                    D_mhalf = spdiags(DCol.^-0.5,0,mFea,mFea);
                    Lt{si} = D_mhalf*Lt{si}*D_mhalf * options.alphaLT;
                    Wt{si} = D_mhalf*Wt{si}*D_mhalf * options.alphaLT;
                    Dt{si} = D_mhalf*Dt{si}*D_mhalf * options.alphaLT;
                end
            else
                Dt{si} = [];
                Lt{si} = [];
            end
    
            Method_Choice='Trim_WA';
            switch Method_Choice
                case 'Original'
                    % initialization (old)
                    V{si} = initV;
                    miv = max(V{si});
                    trimInd = V{si}./max(repmat(miv,size(V{si},1),1),eps) < 5e-2;
                    V{si}(trimInd) = 0;
                    U{si} = backNMF_u(sbjData{si}, K, options, [], V{si});
                case 'Least_Square_Shift'
                    V{si} = initV;
                    U{si} = sbjData{si}*initV*pinv(initV'*initV);
                    U{si} = dataPrepro(U{si},'vp','vmax');
                    U{si} = max(U{si},eps);
                case 'Trim_Least_Square_Truncate'
                    V{si} = initV;
                    miv = max(V{si});
                    trimInd = V{si}./max(repmat(miv,size(V{si},1),1),eps) < 5e-2;
                    V{si}(trimInd) = 0;
                    U{si} = sbjData{si}*initV*pinv(initV'*initV);
                    U{si} = max(U{si},0.01);
                case 'Trim_Least_Square_Exp'
                    V{si} = initV;
                    miv = max(V{si});
                    trimInd = V{si}./max(repmat(miv,size(V{si},1),1),eps) < 5e-2;
                    V{si}(trimInd) = 0;
                    U{si} = sbjData{si}*initV*pinv(initV'*initV);
                    U{si} = exp(U{si})/exp(1);
                case 'Least_Square_Exp'
                    V{si} = initV;
                    U{si} = sbjData{si}*initV*pinv(initV'*initV);
                    U{si} = exp(U{si})/exp(1);
                case 'WA'
                    V{si} = initV;
                    U{si} = sbjData{si}*initV./repmat(sum(initV,1),[size(sbjData{si},1),1]);
                case 'Trim_WA'
                    V{si} = initV;
                    miv = max(V{si});
                    trimInd = V{si}./max(repmat(miv,size(V{si},1),1),eps) < 5e-2;
                    V{si}(trimInd) = 0;
                    U{si} = sbjData{si}*initV./repmat(sum(initV,1),[size(sbjData{si},1),1]);
                case 'Random'
                    V{si} = initV;
                    U{si} = rand(mFea,size(V{si},2));
            end

            Method_Choice='Update_U';
            switch Method_Choice
                case 'Original'
                case 'Update_U'
                    U{si} = backNMF_u(sbjData{si}, K, options, U{si}, V{si});
            end
    
        end
        fprintf('\n');
        initUvName = [resDir,filesep,'init_UV.mat'];
        save(initUvName,'U','V','-v7.3');
        save([outDir '/Preprocessing.mat'], 'W', 'D', 'L', 'Wt', 'Dt', 'Lt', 'U', 'V', '-v7.3');
        %system(['rm ' outDir '/sbjData.mat']);
        pause(3);
        %save([outDir '/sbjData.mat'], 'sbjData', '-v7.3');
    else
        load([outDir '/Preprocessing.mat']);
        %load([outDir '/sbjData.mat']);
    end
    
    disp('------------------');
    % decomposition l21
    [U, V] = mMultiNMF_l21p1_ard(sbjData, W, D, L, Wt, Dt, Lt, U, V, initV, options, outDir);
    
    % finalize the results
    finalUvName = [resDir,filesep,'final_UV.mat'];
    save(finalUvName,'U','V','-v7.3');
    
    % compute the V_centroid
    V_centroid = zeros(size(V{1}));
    for vi=1:length(V)
        V_centroid = V_centroid + V{vi};
    end
    V_centroid = V_centroid / length(V);
    
    outName_cen = [resDir,filesep,'res_cen.mat'];
    save(outName_cen,'V_centroid','-v7.3');
    
    if calcGrp==1
        disp('calculate time course based on group spatial maps (for comparison use later)...');
    
        grpOpt = [];
        grpOpt.maxIter = 500;
        grpOpt.error = 1e-4;
        grpOpt.nRepeat = 1;
        grpOpt.minIter = 100;
        grpOpt.meanFitRatio = 0.1;
        grpOpt.NormW = 1;
    
        bSbjNum = length(V);
        U = cell(bSbjNum,1);
        V = cell(bSbjNum,1);
        for si=1:bSbjNum
            [U{si},~] = backNMF_u(sbjData{si}, K, grpOpt, [], V_centroid);
            V{si} = V_centroid;
        end
    
        outName_grp = [resDir,filesep,'grp_UV.mat'];
        save(outName_grp,'U','V','-v7.3');
    end
    
    end
    
    function [U, V, lambdas, iterLog] = mMultiNMF_l21p1_ard(X, W, D, L, Wt, Dt, Lt, initU, initV, GroupV, options, outDir)
    % with ARD regularization on U and V
    % Modified by Yuncong Ma, 11/30/2022
    % Added group V into update of U and V
    
    viewNum = length(X); % GroupV will always be treated as the first element of V
    
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
        % clear initV;
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
    
    Flag_QC=0;
    UV_Old={U,V};
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
    
        % viewNum=1
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
    
            % precomputed terms for GroupV
            Method_Choice='Original';
            switch Method_Choice
                case 'Original'
                    V0_2=0;
                case 'GroupV'
                    V0_2=max(GroupV.^2,eps);
            end
    
            % Stepsize
            Method_Choice='Original';
            switch Method_Choice
                case 'Original'
                    Step_Size=1;
                case '0.1'
                    Step_Size=0.1;
            end
    
    
            while iter<maxInIter
                iter = iter + 1;
    
                % ===================== update V ========================
                % Eq. 8-11
                % add GroupV
                XU = X{i}'*U{i};
                UU = U{i}'*U{i};
                VUU = V{i}*UU;
    
                tmpl2 = fixl2 + V{i}.^2 + V0_2;
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
    
                if Step_Size==1
                    V{i} = V{i}.*(XU./max(VUU,eps));
                else
                    V{i} = V{i}*(1-Step_Size)+Step_Size*V{i}.*(XU./max(VUU,eps));
                end
    
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
    
                if Step_Size==1
                    U{i} = U{i}.*(XV./max(UVV,eps));
                else
                    U{i} = U{i}*(1-Step_Size)+Step_Size*U{i}.*(XV./max(UVV,eps));
                end
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
    
                % QC Control
                Method_Choice='QC';
                switch Method_Choice
                    case 'Original'
                    case 'QC'
                        temp=corr(GroupV,V{i});
                        QC.Spatial_Correspondence=diag(temp);
                        temp=temp-diag(diag(temp));
                        QC.Spatial_Correspondence_Control=max(temp,[],2);
                        QC.Delta_Sim=min(QC.Spatial_Correspondence-QC.Spatial_Correspondence_Control);
                        if QC.Delta_Sim<=0
                            U=UV_Old{1};
                            V=UV_Old{2};
                            Flag_QC=1;
                            fprintf('\nMeet QC constraint: Delta sim = %f\n',QC.Delta_Sim);
                            break;
                        else
                            fprintf('Delta sim = %f\n',QC.Delta_Sim);
                        end
                end
                UV_Old={U,V};
            end
            if Flag_QC==1
                break
            end
        end
        save([outDir '/Iteration_res.mat'], 'W', 'D', 'L', 'Wt', 'Dt', 'Lt', 'U', 'V', 'LDf', 'Flag_QC', '-v7.3');
        JobStatus = ['Iteration_' num2str(j)];
        save([outDir '/JobStatus.mat'], 'JobStatus');
        if Flag_QC==1
            break
        end
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
    