clc
clear
seed=1;
Point_num_4=250;
% 生成A的正类数据
Amean_pos1 =2 ;%round(rand(1,1)*10); % 正态分布的均值
Astd_pos1 =8;%round(rand(1,1)*10); % 正态分布的标准差
Anum_pos1 = Point_num_4; % 正类数据点的数量
Apos_data1 = normrnd(Amean_pos1, Astd_pos1, Anum_pos1, 2); % 生成正类数据

% 生成A的负类数据
Amean_pos2 =8 ;%round(rand(1,1)*10); % 正态分布的均值
Astd_pos2 =8 ;%round(rand(1,1)*10); % 正态分布的标准差
Anum_pos2 = Point_num_4; % 正类数据点的数量
Apos_data2 = normrnd(Amean_pos2, Astd_pos2, Anum_pos2, 2); % 生成负类数据

% 生成T的正类数据
Tmean_pos1 = 2; % 正态分布的均值
Tstd_pos1 = 10; % 正态分布的标准差
Tnum_pos1 = Point_num_4; % 负类数据点的数量
Tneg_data1 = normrnd(Tmean_pos1, Tstd_pos1, Tnum_pos1, 2); % 生成负类数据

% 生成T的负类数据
Tmean_pos2 = 4; % 正态分布的均值
Tstd_pos2 =10; % 正态分布的标准差
Tnum_pos2 = Point_num_4; % 负类数据点的数量
Tneg_data2 = normrnd(Tmean_pos2,Tstd_pos2, Tnum_pos2, 2); % 生成负类数据

X_train_A=[Apos_data1;Apos_data2];
Y_train_A=[ones(size(Apos_data1,1),1);zeros(size(Apos_data2,1),1)];

X=[Apos_data1;Apos_data2];
Y=[ones(size(Apos_data1,1),1);-ones(size(Apos_data2,1),1)];

X_test=[Tneg_data1;Tneg_data2];
Y_test=[ones(size(Tneg_data1,1),1);zeros(size(Tneg_data2,1),1)];

% 绘制散点图
figure;
scatter(Apos_data1(:, 1),Apos_data1(:, 2), 'r', '^'); % 源域正类数据用红色圆点表示
hold on;
scatter(Apos_data2(:, 1), Apos_data2(:, 2), 'r', 's'); % 源域负类类数据红色叉用表示
hold on;
scatter(Tneg_data1(:, 1), Tneg_data1(:, 2), 'b', 'o'); % 目标域负类数据用蓝色圆点表示
hold on;
scatter(Tneg_data2(:, 1), Tneg_data2(:, 2), 'b', 'x'); % 目标域负类数据用蓝色叉点表示
hold off;
title('Different mean & different Std');
xlabel('x1');
ylabel('x2');
legend('A-Postive', 'A-Negtive','T-Postive', 'T-Negtive');