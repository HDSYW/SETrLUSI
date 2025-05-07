function svm = train_svm(train_data,train_label,kertype,kc,C,Y_A)%  ��ߵ����벻���Դ���
n = length(train_label); % ������������
can = kc;
H = (train_label'*train_label).*kernel(train_data,train_data,kertype,can);
lambda=train_label'.*Y_A;
f = -(ones(n,1)-lambda); % ��֤fΪ��������ԭʽ�а���ת�õĲ���
A = [];% ��������Լ��
b = [];% ��������Լ��
Aeq = train_label;beq = 0;% s.t.: aY = 0;
lb = zeros(n,1);
ub = C*ones(n,1);                       %C*ones(n,1); % 0 <= a <= C
a0 = [];    % �Խ���г�ʼ��?
options = optimoptions('quadprog','Display','off');
a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
a(a<1e-6)=0;a(C-a<1e-6)=C;
%e = 1e-10;
sv_index = find(a>0);
svm.a = a(sv_index);
svm.data = train_data(:,sv_index);
svm.label = train_label(:,sv_index);
end