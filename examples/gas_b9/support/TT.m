function [X_train, Y_train, X_test, Y_test] = TT(X, Y, percent)
    unique_classes = unique(Y);
    num_classes = length(unique_classes);

    X_train = [];
    Y_train = [];
    X_test = [];
    Y_test = [];

    for i = 1:num_classes
        class_idx = (Y == unique_classes(i));
        class_data = [X(class_idx, :), Y(class_idx)];

        num_class_samples = size(class_data, 1);
        num_test_samples = floor(percent * num_class_samples);

        idx_test = randperm(num_class_samples, num_test_samples);
        test_data = class_data(idx_test, :);

        train_data = class_data;
        train_data(idx_test, :) = [];

        X_train = [X_train; train_data(:, 1:end-1)];
        Y_train = [Y_train; train_data(:, end)];
        X_test = [X_test; test_data(:, 1:end-1)];
        Y_test = [Y_test; test_data(:, end)];
    end
end
