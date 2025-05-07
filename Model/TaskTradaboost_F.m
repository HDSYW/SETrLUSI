function [result] = TaskTradaboost_F(Data,ktype,pa)
M=pa.T;gamma=0;
% ---------- Data process ----------
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
Y_test_T(Y_test_T==0)=-1;
Y_train_T(Y_train_T==0)=-1;
if iscell(Data.X_train_A)
    for Dsnum=1:numel(Data.X_train_A)
        data_ori{Dsnum}.X_train_A=Data.X_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A=Data.Y_train_A{Dsnum};
        data_ori{Dsnum}.Y_train_A(data_ori{Dsnum}.Y_train_A==0)=-1;
    end
else
    data_ori{1}.X_train_A=Data.X_train_A;
    data_ori{1}.Y_train_A=Data.Y_train_A;
    data_ori{1}.Y_train_A(data_ori{1}.Y_train_A==0)=-1;
end
best_error_test=1;
for p1=0%pa.min:pa.step:pa.max
    fprintf('--------------------*Regular=%.2f*--------------------\n',p1);
    Para.p1 = 2.^p1;  Para.kpar.kp1 =2.^0; Para.kpar.kp2 = 0;Para.kpar.ktype = ktype;
    [decision_value,H,Ypre_trian_T]  = PhaseI_TaskTrAdaBoost( data_ori,X_train_T,X_test_T,M, gamma,Para);
    n= size(X_train_T,1);
    wT = ones(n, 1) / n;  % Initialize weight vector to uniform distribution
    Alpha_F=zeros(size(X_test_T,1),1);
    Alpha_DV=zeros(size(X_test_T,1),1);

    for t = 1:M
        F = {}; F_train={}; DV={}; E=[];
        PT = wT / sum(wT);  % Normalize the weight vector
        for h = 1:length(H)
            Pre_Train_T = Ypre_trian_T{h};
            Pre_Test=H{h};
            Pre_Test_DV=decision_value{h};
            epsilon(t) = sum(PT.*(Pre_Train_T~=Y_train_T))./sum(PT);

            if epsilon(t) > 0.5
                Ypre_trian_T{h} = -Ypre_trian_T{h};  % Invert the classifier
                Pre_Test_DV=-(Pre_Test_DV);
                Pre_Train_T=-(Pre_Train_T);
                epsilon(t) = 1 - epsilon(t);
            end
            F{end+1} = Pre_Test;
            F_train{end+1}=Pre_Train_T;
            DV{end+1}=Pre_Test_DV;
            E(end+1)=epsilon(t);
        end
        % Find the weak classifier with minimum error
        [et,index]=min(E);
        F_min=F{index};
        F_train_min= F_train{index};
        DV_min=DV{index};
        alpha_t = 0.5 * log((1 - et) / et);

        Alpha_DV= Alpha_DV+alpha_t.*DV_min;
        Alpha_F=Alpha_F+alpha_t.*F_min;

        H(index) = [];  % Remove the selected classifier from H
        
        wT = wT .* exp(-alpha_t *Y_train_T.*F_train_min);% Update the weights

        CM_test = ConfusionMatrix( sign(Alpha_F),Y_test_T);
        Test_error(t)=1-CM_test.Ac./100;
        % ---------- Print result ----------
        fprintf('Iteration %s \t',num2str(t))
        fprintf('Train_error=%.4f \t',et)
        fprintf('Test_error=%.4f\t\n',Test_error(t))
    end
    if Test_error(pa.T)<=best_error_test
        best_error_test_H=Test_error;
        best_error_test= Test_error(pa.T);
        best_Final_Y_test=Alpha_DV;
        best_Ypre_test= sign(Alpha_F);
        best_regular=p1;
    end
    clear bestTestPredict bestTtrainPredict alpha er bestvalue Test_error
end
% ---------- Figure ----------
if sum(pa.figure=='On')
    figure
    x=1:pa.T;
    y_1=er;
    y_2=Test_error;
    plot(x,y_1,"Marker",".","MarkerSize",15,Color=[223/255,122/255,094/255])
    hold on
    plot(x,y_2,"Marker",".","MarkerSize",15,Color=[060/255,064/255,091/255])
    box off
    grid on
    grid minor
    legend("TrainError","TestError")
    title("MutiTradaboost")
    drawnow;
end


% ---------- Output ----------
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, best_Final_Y_test, '1');
result.AUC=100*AUC;
result.testerror=best_error_test_H;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));
end

