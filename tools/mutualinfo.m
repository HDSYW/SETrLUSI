function averageMI = mutualinfo(X, Y, numBins)
    % 计算矩阵 X 和 Y 中每行之间的互信息
    % X, Y: 输入矩阵，大小为 (100, 5)
    % numBins: 用于计算直方图的分箱数量
    % 返回：一个包含每行互信息的向量 MI

    % 确保输入矩阵大小为 (100, 5)
    [numRowsX, numColsX] = size(X);
    [numRowsY, numColsY] = size(Y);

    % 初始化互信息向量
    MI = zeros(numRowsX, 1);
    
    % 遍历每一行，计算互信息
    for i = 1:numRowsX
        for j=1:numRowsY
         MI(i,j) = mutualInformation(X(i, :), Y(j, :), numBins);
        end
    end
    averageMI=mean(mean(MI));
end

function MI = mutualInformation(X1, X2, numBins)
    % 计算X1和X2之间的互信息
    % X1, X2: 输入向量
    % numBins: 用于计算直方图的分箱数量

    % 确保输入向量是列向量
    X1 = X1(:);
    X2 = X2(:);

    % 联合直方图
    jointHist = histcounts2(X1, X2, numBins);

    % 归一化联合直方图得到联合概率分布
    jointProb = jointHist / sum(jointHist(:));

    % 计算边缘直方图和边缘概率分布
    marginalProbX1 = sum(jointProb, 2); % X1 的边缘概率分布
    marginalProbX2 = sum(jointProb, 1); % X2 的边缘概率分布

    % 初始化互信息
    MI = 0;

    % 计算互信息
    for i = 1:numBins
        for j = 1:numBins
            if jointProb(i, j) > 0
                MI = MI + jointProb(i, j) * log2(jointProb(i, j) / (marginalProbX1(i) * marginalProbX2(j)));
            end
        end
    end
end


