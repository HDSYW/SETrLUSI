function bootstrapSamples = bootstraping(data, numDataPoints)
% 输入参数:
% - data: 原始数据集，每行一个样本
% - numSamples: 拔靴法抽样的次数

% 获取原始数据集的样本数量
numDataPoints = size(data, 1);

% 执行拔靴法抽样
% 随机抽取有放回的样本，构建一个重抽样数据集
resampledIndices = randi(numDataPoints, numDataPoints, 1);
resampledData = data(resampledIndices, :);
% 存储重抽样数据集
bootstrapSamples = resampledData;
end


