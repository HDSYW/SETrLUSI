clc;
clear;
%% ���ݵ���
load heart
[X,PS]=mapminmax(X,0,1);
X=mapminmax(X',0,1)';%��X���й�һ������
%% ������֤
k = 10;%
indices = crossvalind('Kfold',X(:,1),k); %����k�۷���������ֵ
%%  ģ��ѵ��
ac=[];
best_C=0;
kc=0;
AC=0;
kernel='linear';
    for C=1:10
        for i = 1:k
            test = (indices == i); train = ~test;
            DataTrain.X = X(train,:);
            DataTrain.Y = Y(train,:);
            DataTest.X = X(test,:);
            DataTest.Y = Y(test,:);
            svm=train_svm(DataTrain.X',DataTrain.Y',kernel,kc,1);
            [result,wb]=test_svm(svm,DataTest.X',DataTest.Y',kernel,kc);
            ac(i)=1-result.ero;
        end
            fprintf('��%d��',C)
            fprintf('׼ȷ�ʣ�%f\n',mean(ac))
            fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if AC<mean(ac)
            AC=mean(ac);
            best_C=C;
        end
    end
fprintf('���׼ȷ�ʣ�%f\n',AC);
fprintf('C��%f\n',best_C);
fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')