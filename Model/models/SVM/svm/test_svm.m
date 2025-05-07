function [result,wb] = test_svm(svm,test_data,test_label,kerneltype,kc,f_A)
%X = DataTest.X;
%tlabel = DataTest.Y;% input:
% svm: train_svm�������ص�֧������
% test_data: ��������
% test_label���Լ���ǩ
% kerneltype�ͼ������࣬��ʽ��������ѡ��inear gaussian mullinear
% output:
% result.Y:���Լ��е�Ԥ�����? result.Y ��[+1,-1]
% result.accuracy:���Լ���׼ȷ��
% b���󷨣�ѡ�����������0�� ai ��C���bi������b����ȡƽ������

sum_b = svm.label - (svm.a'.* svm.label)*kernel(svm.data,svm.data,kerneltype,kc); % bj = yj-sum(ai*yi*<xi,xj>)
b = mean(sum_b);
wx = (svm.a'.* svm.label)*kernel(svm.data,test_data,kerneltype,kc);% w = sum(ai*yi*K(x,xi))

%result.y = wx+b;      %fx
result.Y = sign(wx+b+f_A');% ����
result.ero = 1-length(find(result.Y==test_label))/length(test_label);%mmΪ��ȷ����ĸ���

% result.accuracy = length(find(result.Y==test_label))/length(test_label);% Ԥ��׼ȷ��
% index = find(result.Y==1);
% result.Y1 = result.Y(index);
% test_label1 = test_label(index);
% result.pr =length(find(result.Y1==test_label1)) /length(result.Y>0);%pr
wb.B = b;
wb.wx=wx;
end