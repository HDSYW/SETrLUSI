function S_prime = Layer1_MHTL_AdaBoost(S, Tu, k)
    % Input: S (source domain data sets), Tu (target domain data sets), k (number of clusters)
    % Output: S_prime (labeled source data S')

    % Step 1: Initialize
    [n, ~] = size(Tu);
    indices = randperm(n, k);
    mu = Tu(indices, :);  % Initial mean vector
    C = cell(k, 1);  % Initial clusters
    S_prime = [];

    % Step 2: Cluster assignment
    while true
        % Step 3: Assign each sample in Tu to the nearest mean vector
        for j = 1:n
            d = zeros(k, 1);
            for i = 1:k
                d(i) = norm(Tu(j, :) - mu(i, :),2);
            end
            [~, lambda] = min(d);
            C{lambda} = [C{lambda}; Tu(j, :)];
        end

        % Step 4: Update mean vectors
        mu_new = zeros(k, size(Tu, 2));
        for i = 1:k
            if ~isempty(C{i})
                mu_new(i, :) = mean(C{i}, 1);
            else
                mu_new(i, :) = mu(i, :);  % If cluster is empty, keep the old mean
            end
        end

        % Step 5: Check for convergence
        if all(mu == mu_new)
            break;
        end
        mu = mu_new;
        C = cell(k, 1);  % Reset clusters
    end

    % Step 6: Calculate cluster mean radius and add labeled samples to S_prime
    for i = 1:k
        for j=1:numel(C)
            R = mean(pdist2(C{j}, mu(j, :)));
            for r = 1:size(S, 1)
                if r>=size(S, 1)
                        break
                end
                if norm(S(r, 1:end-1)-mu(i, :),2) < R
                    S_prime = [S_prime; S(r, :)];
                    S(r, :) = [];
                end
            end
            if isempty(S)
              break
            end
        end
    end
end

