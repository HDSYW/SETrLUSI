function f = TELUSI(Ds_old , Dt_train, Dt_test_X, R, H, randomratio, Para)
N=2;% the num of Ds N 
M=5;% the num of predicates M
U=N*M;
% Sampling from Ds and Dt
Dt_train=[Dt_train.X,Dt_train.Y];
Dt_train_new = bootstraping(Dt_train, U,R);
uu=1;
for u=1:U
    for k=1:R
        [~,~,Ds.X, Ds.Y] = TT(Ds_old{uu}.X_train_A,Ds_old{uu}.Y_train_A,randomratio);
        Ds.Y(Ds.Y==-1)=0;
        Ds_new{u,k}= [Ds.X, Ds.Y];
        D_train{u,k}=[Dt_train_new{u, k};Ds_new{u,k}];
        if sum(Para.Vtype=='V')
            [V{u,k},~] = Vmatrix(D_train{u,k},CDFx,Para.v_sig,Para.v_ker); Para.V=V;
        else
            V{u,k}=eye(size(D_train{u,k},1),size(D_train{u,k},1));
        end
        [P{u,k},~] = DPcaculate(D_train{u,k}(:,1:2), D_train{u,k}(:,end), Ds_old{uu}.X_train_A, Para.P_Y, ...
            Ds_old{uu}.Y_train_A, Para.DV, Para.KP, 'X', Para.p3);%大小为D_train x D_train
        clear Ds_new Ds
    end
    if rem(u,10)==6
        uu=uu+1;
    end
end 

% Initialize weights
for numds=1:U/5-1
    Ns = size(D_train{1+numds*5,1}, 1); % Number of source samples
    Nt = size(Dt_train, 1); % Number of target samples
    w_s_init(numds) = 1 / (Ns  + Nt);
    w_t_init = 1 / (Ns  + Nt);
end

for t = 1:H
    F = {}; % Set of candidate weak classifiers
    for  u= 1:U
        for k = 1:R
            % Learn classifier (Placeholder function)
            Para.P=P{u,k};
            Para.V=V{u,k};
            D_training.X=D_train{u,k}(:,1:end-1);
            D_training.Y=D_train{u,k}(:,end);
            [Y,Model] = LUSI_VSVM(Dt_test_X , D_training, Para);
            F{t} = classifier;
            % Compute error of classifier (Placeholder function)
            e_kr = computeError(classifier, Ds, Dt_train, w_s, w_t);

            % Update weights for classifier
            if e_kr < 0.5
                beta_k(k) = e_kr / (1 - e_kr);
                % Update weights (Placeholder functions)
                w_s = updateWeights(w_s, classifier, Ds, beta_k(k));
                w_t = updateWeights(w_t, classifier, Dt_train, beta_k(k));
            else
                beta_k(k) = 0;
            end
        end
    end

    % Compute the error for all classifiers
    e_t = mean(beta_k(beta_k > 0));

    % Update the decision function
    if e_t <= 0.5
        fT(t) = sum(beta_k .* cellfun(@(clf) classify(clf, Dt_train), F)) / sum(beta_k);
    end
end

% Final decision function
f = @(x) sum(fT) > H / 2;
end


