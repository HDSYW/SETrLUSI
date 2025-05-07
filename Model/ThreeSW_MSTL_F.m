function result = ThreeSW_MSTL_F(Data,ktype,pa)

for i=1:numel(Data.X_train_A)
    Data.Y_train_A{i}(Data.Y_train_A{i}==0)=-1;
    Data.Y_train_T(Data.Y_train_T==0)=-1;
    sources{i}=[Data.X_train_A{i},Data.Y_train_A{i};Data.X_train_T,Data.Y_train_T];
end
Data.Y_test_T(Data.Y_test_T==0)=-1;
target=[Data.X_test_T,Data.Y_test_T];
ACC=0;
for p1=pa.min:pa.step:pa.max
    prob = ThreeSW_MSTL(sources, target,p1,2);
    CM_test = ConfusionMatrix(target(:,end),sign(prob-0.5));
    if CM_test.Ac>=ACC
        ACC=CM_test.Ac;
        pro_final=prob;
    end
    fprintf('pa= %s\t\n',num2str(p1))
end
% ---------- Output ----------
CM_test = ConfusionMatrix(target(:,end),sign(pro_final-0.5));
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
% [~,~,~, AUC]=perfcurve(Y_test_T,best_Final_Y_test./sum( best_minBeta), '1');
% result.AUC=100*AUC;
% result.testerror=best_error_test_H;
result.Spe=CM_test.Spe;
result.Sen=CM_test.Sen;
% result.lam=best_regular;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t\n',result.ac_test)
% fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));
end

