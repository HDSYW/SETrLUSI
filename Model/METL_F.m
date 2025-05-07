function [result] = METL_F(Data,ktype,pa)

% ---------- Data process ----------
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
Y_test_T(Y_test_T==0)=-1;Y_train_T(Y_train_T==0)=-1;
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
num_DS=numel(data_ori);
N =  pa.T; % 迭代次数
% Y_train_T(Y_train_T==-1)=2;
DT=[X_train_T,Y_train_T];

fi_ensemble = cell(3, 1); % 存储三个分类器
for o=1:num_DS
    DSi=[data_ori{o}.X_train_A,data_ori{o}.Y_train_A];
    Dn_Si = [DSi; DT]; % 初始合并数据集
    Dn_Si_D = Dn_Si;
    for n = 1:N
        % 训练 Softmax 分类器
        YSoft_train=Dn_Si_D(:, end);
        YSoft_train(YSoft_train==-1)=2;
        %         softmaxModel = trainSoftmaxClassifier(Dn_Si_D(:, 1:end-1),YSoft_train);
        [W, b] = trainSoftmaxClassifier(Dn_Si_D(:, 1:end-1), YSoft_train, 2,10000, 0.5);
        fi_ensemble{1} = [W;b];

        % 训练 SVM 分类器
        ACC=0;
        Ytrain=Dn_Si_D(:, end);
        Ytrain(Ytrain==2)=-1;
        for p1=pa.min:pa.step:pa.max
            svmModel = fitcsvm(Dn_Si_D(:, 1:end-1), Ytrain,"BoxConstraint",2^p1);
            Y=predict(svmModel, X_train_T);
            CM = ConfusionMatrix(Y,Y_train_T);
            if CM.Ac>ACC
                ACC=CM.Ac;
                fi_ensemble{2}=svmModel;
                best_p1(o)=p1;
            end
        end

        % 训练 DNN 分类器
        YDNN_train=Dn_Si_D(:, end);
        YDNN_train(YDNN_train==-1)=2;
        dnnModel = trainDNNClassifier(Dn_Si_D(:, 1:end-1), YDNN_train);
        fi_ensemble{3} = dnnModel;

        % 更新 Dn_Si_D
        Dn_Si_new = [];
        eqe=double(classifyData(fi_ensemble, DSi(:, 1:end-1)));
        for ii = 1:length(DSi(:, end))
            if eqe(ii) == 1
                Dn_Si_new = [Dn_Si_new; DSi(ii, :)];
            end
        end
        if isequal(Dn_Si_new,Dn_Si)
            break
        end
        Dn_Si_D=[Dn_Si_new; DT];
        fprintf('Source domain %s Iteration %s \n',num2str(o),num2str(n))
    end
    w_i(o) = mutualinfo(DSi(:,1:end-1), X_test_T,10);
    prLabels = predictSoftmaxClassifier(X_test_T, fi_ensemble{1}(1:size(X_test_T,2),:),fi_ensemble{1}(end,:));
    prLabels(prLabels==2)=-1;
    Ysvm = predict(fi_ensemble{2}, X_test_T);
    YDNN = double(classify(fi_ensemble{3}, X_test_T));
    YDNN(YDNN==2)=-1;
    f{o}=w_i(o).*((prLabels+YDNN+Ysvm)./3);
end
finalOutput=zeros(length(Y_test_T),1);
for oo=1:num_DS
    finalOutput=finalOutput+f{oo}./sum(w_i);
end
% ---------- Output ----------
best_Ypre_test=sign(finalOutput);
CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t\n',result.ac_test)
fprintf('%s\n', repmat('=', 1, 60));

    function [W, b] = trainSoftmaxClassifier(X, Y, numClasses, numIterations, learningRate)

        [numSamples, numFeatures] = size(X);

        % 初始化权重和偏置
        W = randn(numFeatures, numClasses) * 0.01;
        b = zeros(1, numClasses);

        % 进行迭代训练
        for iter = 1:numIterations
            % 计算分数
            scores = X * W + b;

            % 计算 softmax 概率
            expScores = exp(scores);
            probs = expScores ./ sum(expScores, 2);

            % 计算梯度
            dscores = probs;
            dscores(sub2ind(size(dscores), (1:numSamples)', Y)) = dscores(sub2ind(size(dscores), (1:numSamples)', Y)) - 1;
            dscores = dscores / numSamples;

            dW = X' * dscores;
            db = sum(dscores, 1);

            % 更新权重和偏置
            W = W - learningRate * dW;
            b = b - learningRate * db;
        end
    end

    function predictedLabels = predictSoftmaxClassifier(X, W, b)
        % X: 输入数据矩阵，大小为 (numSamples, numFeatures)
        % W, b: 训练好的权重和偏置

        % 计算分数
        scores = X * W + b;

        % 计算 softmax 概率
        expScores = exp(scores);
        probs = expScores ./ sum(expScores, 2);

        % 预测标签
        [~, predictedLabels] = max(probs, [], 2);
    end

    function model = trainDNNClassifier(X, Y)
        % Train DNN
        layers = [ ...
            featureInputLayer(size(X, 2))
            fullyConnectedLayer(10)
            reluLayer
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer];
        options = trainingOptions('adam','ExecutionEnvironment','gpu', 'MaxEpochs', 500, 'MiniBatchSize', 256, ...
            'InitialLearnRate', 0.01,'Verbose',false,'Plots','none');
        model = trainNetwork(X, categorical(Y), layers, options);
    end

    function eqe = classifyData(models, X)
        Labels= predictSoftmaxClassifier(X, models{1}(1:size(X,2),:),models{1}(end,:));
        Labels( Labels==2)=-1;
        svmLabel = predict(models{2}, X);
        dnnProb = double(classify(models{3}, X));
        dnnProb(dnnProb==2)=-1;
        eqe=(svmLabel==Labels)&(svmLabel==dnnProb);
    end

end

