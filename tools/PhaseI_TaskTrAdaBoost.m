function [DV,H,Ypre_trian_T] = PhaseI_TaskTrAdaBoost(sourceData,X_train_T,X_test_T,M, gamma,Para)
% Input:
% - sourceData: A cell array containing source training data DS1, ..., DSN
% - M: The maximum number of iterations
% - gamma: The regularizing threshold

% Output:
% - H: Set of candidate weak classifiers

H = {};  % Initialize the set of candidate weak classifiers
DV={};
Ypre_trian_T={};
for k = 1:length(sourceData)
    DSk = sourceData{k};
    [n(k), ~] = size(DSk.Y_train_A);
    wSk = ones(n(k), 1) / n(k);  % Initialize weight vector to uniform distribution
    for t = 1:M
        P = wSk / sum(wSk);  % Normalize the weight vector
        X = sourceData{k}.X_train_A; Y = sourceData{k}.Y_train_A;
        idx = randsample(size(X,1), floor(size(X,1)/2), true, P);
        Trn.X = X(idx, :);Trn.Y = Y(idx);
        % Train a weak classifier hkt with weighted data
        [TrainPredict{k} , ~] = LIB_L1SVC(X , Trn , Para);
        [TrainPredict_T{k} , ~] = LIB_L1SVC(X_train_T , Trn , Para);
        [TestPredict{k} , Model{k}] = LIB_L1SVC(X_test_T , Trn , Para);
        % Compute the error of hkt
        epsilon(t) = sum(P.*(TrainPredict{k}~=Y))./sum(P);
        if epsilon(t)==0
            epsilon(t)=0.0001;
        elseif epsilon(t)>=0.5
            epsilon(t)=0.4999;
        end
        % Calculate alpha
        alpha(t) = 0.5 * log((1 - epsilon(t)) / epsilon(t));
        if alpha(t) > gamma
            H{end+1} = TestPredict{k};  % Add to the set of candidate weak classifiers
            DV{end+1} = Model{k}.dv;
            Ypre_trian_T{end+1} = TrainPredict_T{k};
        end
        % Update the weights
        wSk = wSk .* exp(-(alpha(t) *Y.*TrainPredict{k}));
    end
end
end
