function result = SETrLUSI_VSVM_Ram_New(Data,ktype,v_ker,CDFx,pa)
best_error_test=1;
% >>>>>>>>>>>>>>>>>>>>Dataset<<<<<<<<<<<<<<<<<<<<
X_train_T=Data.X_train_T;Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;Y_test_T=Data.Y_test_T;Y_test_T(Y_test_T==-1)=0;
if iscell(Data.X_train_A)
    for Dsnum=1:numel(Data.X_train_A)
        data_ori{Dsnum}.X_train_A=Data.X_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A=Data.Y_train_A{Dsnum};
    end
else
    data_ori{1}.X_train_A=Data.X_train_A;
    data_ori{1}.Y_train_A=Data.Y_train_A;
    data_ori{1}.Y_train_A(data_ori{1}.Y_train_A==0)=-1;
end
Y_train_T(Y_train_T==-1)=0;
for RD=0.1:0.2:0.9
    pa.randomratio=RD;
    for p3=0.1:0.2:0.9
        for p1=pa.min:pa.step:pa.max
            % ---------- Para change ----------
            Para.kpar.kp1 =2.^0; Para.p1=2.^p1; Para.kpar.kp2 = 0; Para.v_sig = 2.^0;Para.p3=p3;
            % >>>>>>>>>>>>>>>>>>>>Para set<<<<<<<<<<<<<<<<<<<<
            V_Matrix = 'Vmatrix'; V_MatrixFun = str2func(V_Matrix);
            Para.vmatrix = V_Matrix; Para.CDFx = CDFx; Para.v_ker = v_ker; CDFx = Para.CDFx;
            Para.kpar.ktype = ktype; Para.KP.ktype=Para.kpar.ktype; Para.KP.kp1=Para.kpar.kp1;
            Para_svm.kpar.ktype = ktype; Para_svm.kpar.kp1 =2.^-8; Para_svm.kpar.kp2 = 0;
            %% >>>>>>>>>>>>>>>>>>>> Learning <<<<<<<<<<<<<<<<<<<<
            % ---------- Predict predicate ----------
            Para.Vtype= pa.Vtype; predicate_ori=pa.predicate_ori;
            if RD==0.1 && p3==0.1 && p1==pa.min
                fprintf('*Vtype=%s* *Ptype=', Para.Vtype)
                for i = 1:length(predicate_ori)
                    if i < length(predicate_ori)
                        fprintf('%s | ', predicate_ori(i));
                    else
                        fprintf('%s*\n', predicate_ori(i));
                    end
                end
            end
            %     fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
            N=numel(data_ori); % the num of Ds N
            U=length(predicate_ori); % the num of predicates M
            % ---------- Sampling from Ds and Dt ----------
            Dt_train=[X_train_T,Y_train_T];
            for h=1:pa.H % The num of iterations
                Final_Y_test=zeros(size(X_test_T,1),1);
                for i=1:N
                    if h==1
                        w_t{i}=ones(size(Y_train_T,1),1);
                    end
                    index{i}=randi(U);
                    if sum(pa.ram=='Open P')
                        Ds_new{i}= [data_ori{i}.X_train_A,data_ori{i}.Y_train_A];
                        Dt_train_new{i} =Dt_train;
                        P{i}=rand(1,size(Dt_train_new{i},1))';
                    elseif sum(pa.ram=='Open DSDT')
                        [~,~,Ds.X, Ds.Y] = TT(data_ori{i}.X_train_A,data_ori{i}.Y_train_A,pa.randomratio); Ds.Y(Ds.Y==-1)=0;
                        Ds_new{i}= [Ds.X, Ds.Y];
                        Dt_train_new{i} = bootstraping(Dt_train, size(Y_train_T,1));
                        P{i}=rand(1,size(Dt_train_new{i},1))';
                    elseif sum(pa.ram=='All Close')
                        Ds_new{i}= [data_ori{i}.X_train_A,data_ori{i}.Y_train_A];
                        Dt_train_new{i} =Dt_train;
                        P{i}=w_t{i}./sum(w_t{i});
                    elseif sum(pa.ram=='All Open')
                        [~,~,Ds.X, Ds.Y] = TT(data_ori{i}.X_train_A,data_ori{i}.Y_train_A,pa.randomratio); Ds.Y(Ds.Y==-1)=0;
                        Ds_new{i}= [Ds.X, Ds.Y];
                        Dt_train_new{i} = bootstraping(Dt_train, size(Y_train_T,1));
                        P{i}=rand(1,size(Dt_train_new{i},1))';
                    end

                    idx = randsample(size(Dt_train_new{i},1), floor(size(Dt_train_new{i},1)/2), true, P{i}');

                    Data_train.X=Dt_train_new{i}(idx,1:end-1); Data_train.Y=Dt_train_new{i}(idx,end);

                    DS.X=Ds_new{i}(:,1:end-1); DS.Y=Ds_new{i}(:,end);

                    % Ramdon SVM
                    Para_svm.p1 = 2.^(randi([-8,8]));
                    if sum(predicate_ori(index{i})=='A_Y') || sum(predicate_ori(index{i})=='DV')
                        model = lintrain( DS.Y, sparse(DS.X),'-q');
                        [Para.A_Y, accuracy, Para.DV]=linpredict(Data_train.Y,sparse(Data_train.X),model,'-q');
%                         [Para.A_Y , model] = LIB_L1SVC(Data_train.X , DS , Para_svm);
%                         Para.DV=model.dv;
                    else
                        Para.A_Y=1;Para.DV=0;
                    end

                    X_train_T_new=Dt_train_new{i}(:,1:end-1);
                    Y_train_T_new=Dt_train_new{i}(:,end);
                    % ---------- Calculate V & P----------
                    if sum(Para.Vtype=='V')
                        [Para.V,~] = Vmatrix(Data_train.X,CDFx,Para.v_sig,Para.v_ker);
                    else
                        Para.V=eye(size(Data_train.X,1),size(Data_train.X,1));
                    end
                    [Para.P,~] = DPcaculate(Data_train.X, Data_train.Y, Ds_new{i}(:,1:end-1), Para.A_Y, ...
                        Ds_new{i}(:,end), Para.DV, Para.KP, predicate_ori(index{i}), Para.p3);
                    if sum(pa.ram=='All Close') || sum(pa.ram=='Open DSDT')
                        Para.p3=0;
                    end
                    % ---------- Model training----------
                    [Y,Model] = LUSI_VSVM(X_test_T , Data_train, Para); Y.tst(Y.tst==-1)=0;Y.trn(Y.trn==-1)=0;
                    [Y_valx,~] = LUSI_VSVM(X_train_T_new, Data_train, Para); Y_valx.tst(Y_valx.tst==-1)=0;Y_valx.trn(Y_valx.trn==-1)=0;
                    % ---------- Calculate error on Dt and Update weights for classifier ----------
                    error = sum((Y_valx.tst~=Y_train_T_new))./length(Y_valx.tst);
                    if error < 0.50
                        e_kr(i)=error;
                    else
                        e_kr(i)=0.499;
                    end
                    if error==0
                        e_kr(i)=0.001;
                    end
                    beta_kt(i) = 1-e_kr(i)/(1 - e_kr(i));
                    for m=1:size(Y_train_T_new,1)
                        w_t{i}(m) = w_t{i}(m).*(beta_kt(i)^(-abs((Y_valx.tst(m)-Y_train_T_new(m)))));
                    end
                    beta_Y_test{i,h}= beta_kt(i).*Model.f;
                end
                % ---------- Ensemble result of u-th predicate in h-th iteration ----------
                [~,ErIndex(h)]=min(e_kr);
                minBeta(h)=beta_kt(ErIndex(h));
                if min(e_kr)==0.499
                    beta_Y_test{ErIndex(h),h}=zeros(length(Model.f),1);
                    minBeta(h)=0;
                end
                minpredicate{h}=predicate_ori(index{ErIndex(h)});
                for ii=1:h
                    Final_Y_test=Final_Y_test+beta_Y_test{ErIndex(ii),ii};
                end
                Ypre_test=(sign(Final_Y_test./sum(minBeta)-0.5)+1)./2;
                final_error_test(h)=sum(abs(Ypre_test-Y_test_T))./size(X_test_T,1);
            end
            if final_error_test(pa.H)<=best_error_test
                best_error_test_H=final_error_test;
                best_error_test= final_error_test(pa.H);
                best_Final_Y_test=Final_Y_test;
                best_Ypre_test= Ypre_test;
                best_minBeta=minBeta;
                best_regular=p1;
                best_p3=Para.p3;
                best_RD=RD;
            end
            folderpath=pa.DA+'/TE/';
            mkdir(folderpath);
            filename=sprintf('file_%d.mat',best_regular);
            fullpath=fullfile(folderpath,filename);
            save(fullpath,'best_error_test_H')
            clear minBeta
        end
    end
    fprintf('Complete %s\t\n',num2str((RD)))
end
% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
[~,~,~, AUC]=perfcurve(Y_test_T,best_Final_Y_test./sum( best_minBeta), '1');
result.AUC=100*AUC;
result.testerror=best_error_test_H;
result.Spe=CM_test.Spe;
result.Sen=CM_test.Sen;
result.lam=best_regular;
result.p3=best_p3;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t',best_regular)
fprintf('Finall_p1=%.4f\t',best_p3)
fprintf('Finall_RD=%.4f\t\n',best_RD)
fprintf('%s\n', repmat('=', 1, 60));
end

