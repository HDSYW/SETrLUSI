function YT_hat = master_F(Data, ktype,pa)
lambda = 0.5; % default value for lambda
% ---------- Data process ----------
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
XT=Data.X_test_T;
Y_test_T=Data.Y_test_T;
YT_hat = zeros(size(XT, 1), 1);
Y_test_T(Y_test_T==0)=-1;Y_train_T(Y_train_T==0)=-1;
if iscell(Data.X_train_A)
    for Dsnum=1:numel(Data.X_train_A)
        sourceDatasets{Dsnum}.X=Data.X_train_A{Dsnum};
        sourceDatasets{Dsnum}.Y=Data.Y_train_A{Dsnum};
        sourceDatasets{Dsnum}.Y(sourceDatasets{Dsnum}.Y==0)=-1;
    end
else
    sourceDatasets{1}.X=Data.X_train_A;
    sourceDatasets{1}.Y=Data.Y_train_A;
    sourceDatasets{1}.Y(sourceDatasets{1}.Y==0)=-1;
end
% sourceDatasets: cell array containing N source datasets, each cell is a struct with fields X and Y
% targetDataset: struct with fields X and Y (unlabeled data, Y might be empty)
% lambda: hyper-parameter

% Initialize variables


% Main algorithm loop
for DSk_idx = 1:length(sourceDatasets)
    DSk = sourceDatasets{DSk_idx};
    % Step 3: Cal
    for i=1:size(DSk.X,2)
        NI(i)=mutualInformation(DSk.X(:,i), DSk.Y,length(DSk.Y),length(DSk.Y));
        NKL(i)=KL(DSk.X, XT,i, i, length(DSk.Y));
        theta(i)=computeMIC(DSk.X(:,i), DSk.Y);
    end
    NI=NI./sum(NI);
    NKL=NKL./sum(NKL);
    w(DSk_idx)=sum(1./(1+exp(-lambda*NI+(1-lambda)*NKL)));
    theta=theta./sum(theta);% feature important
    max_x=max(XT);
    min_x=min(XT);
    % Step 6: Calculate feature-weighted similarity and instance weight
    for xi_idx = 1:size(DSk.X, 1)
        xi = DSk.X(xi_idx, :);
        fws(xi_idx) =sum(double((max_x>=xi) & (xi>=min_x)).*theta);
        instance_weight(xi_idx) = fws(xi_idx)./((sum(theta)-fws(xi_idx)+1)^2);
    end
    % Step 7: Calculate prior probability of each class
    prior_prob_1 = (sum(fws*double(DSk.Y==1))+1)./(sum(fws)+sum(DSk.Y==1));
    prior_prob_0 = (sum(fws*double(DSk.Y==-1))+1)./(sum(fws)+sum(DSk.Y==-1));
    % Step 8: Calculate class conditional probability
    numerator1=0;
    denominator1=0;
    numerator0=0;
    denominator0=0;
    for j = 1:size(XT,1)
        x_tj=XT(j,:);
        for k = 1:size(DSk.X,2)
            for i = 1:length(DSk.Y)
                x_si=DSk.X(i,:);
                numerator1 = numerator1+fws(i) .* (x_si(:,k)==x_tj(:,k)) .* (DSk.Y(i)==1);
                denominator1 =  denominator1+fws(i) .* (DSk.Y(i)==1) ;
                numerator0 = numerator0+fws(i) .* (x_si(:,k)==x_tj(:,k)) .* (DSk.Y(i)==-1);
                denominator0 =  denominator0+fws(i) .* (DSk.Y(i)==-1) ;
            end
            p_j1(k)=((numerator1+1)./(denominator1+ length(unique(DSk.X(:,k)))))^(exp(theta(k)));
            p_j0(k)=((numerator0+1)./(denominator0+ length(unique(DSk.X(:,k)))))^(exp(theta(k)));
        end
        sum_p1(j)=sum(p_j1);sum_p0(j)=sum(p_j0);
    end
    h1=(prior_prob_1.*sum_p1)./((prior_prob_1.*sum_p1)+(prior_prob_0.*sum_p0));
    h0=(prior_prob_1.*sum_p1)./((prior_prob_1.*sum_p1)+(prior_prob_0.*sum_p0));
    Ypre{DSk_idx}=double(h1>=h0)';
end
best_Final_Y_test=0;
for ii=1:length(sourceDatasets)
    best_Final_Y_test=best_Final_Y_test+w(ii).*Ypre{ii}./sum(w);
end
best_Ypre_test=sign(best_Final_Y_test);

CM_test = ConfusionMatrix(best_Ypre_test,Y_test_T);
result.ac_test=CM_test.Ac;
result.F=CM_test.FM;
result.GM=CM_test.GM;
result.testerror=0;
[~,~,~, AUC]=perfcurve(Y_test_T, sigmoid(best_Final_Y_test), '1');
result.AUC=100*AUC;
result.Spe=CM_test.Spe; result.Sen=CM_test.Sen;
result.lam=0;
fprintf('%s\n', repmat('-', 1, 60));
fprintf('Finall_Test_AC=%.4f\t',result.ac_test)
fprintf('Finall_regular=%.4f\t\n',best_regular)
fprintf('%s\n', repmat('=', 1, 60));

    function mic_value = computeMIC(X, Y)
    % Ensure the inputs are column vectors
    X = X(:);
    Y = Y(:);

    % Parameters
    alpha = 0.6;   % Control the grid resolution
    c = 15;       % Control the number of grid resolutions

    % Initialize variables
    n = length(X);
    B = floor(n^alpha);

    max_mi = 0;

    % Loop through all possible grid resolutions
    for x_bins = 2:B
        for y_bins = 2:B
            % Calculate mutual information for the given grid resolution
            mi = mutualInformation(X, Y, x_bins, y_bins);
            if mi > max_mi
                max_mi = mi;
            end
        end
    end

    % Normalize the mutual information
    mic_value = max_mi / log(min(B, c));
    end

    function mi = mutualInformation(X, Y, x_bins, y_bins)
    % Discretize the data
    [~, ~, x_bin_indices] = histcounts(X, x_bins);
    [~, ~, y_bin_indices] = histcounts(Y, y_bins);

    % Joint probability distribution
    joint_prob = accumarray([x_bin_indices, y_bin_indices], 1, [x_bins, y_bins]) / length(X);

    % Marginal probability distributions
    x_prob = sum(joint_prob, 2);
    y_prob = sum(joint_prob, 1);

    % Calculate mutual information
    mi = 0;
    for iii = 1:x_bins
        for jjj = 1:y_bins
            if joint_prob(iii, jjj) > 0
                mi = mi + joint_prob(iii, jjj) * log(joint_prob(iii, jjj) / (x_prob(iii) * y_prob(jjj)));
            end
        end
    end
    end
end
