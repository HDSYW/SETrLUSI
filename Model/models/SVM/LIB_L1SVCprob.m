function [ PredictY , model ] = LIB_L1SVCprob( ValX , Trn , Para )
% min  0.5*norm(w,2)^2 + C*e'*xi , 
%  s.t .  Y.*(X*w+e*b) >= e - xi,    xi >= 0. 
% Solvin Hinge SVC with LIBSVM tool box. 
% Ref - C.-C. Chang and C.-J. Lin. LIBSVM : a library for support vector machines. 
%          ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
% Site - https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% _______________________________ Input  _______________________________
%      Trn.X  -  m x n matrix, explanatory variables in training data 
%      Trn.Y  -  m x 1 vector, response variables in training data 
%      ValX   -  mt x n matrix, explanatory variables in Validation data 
%      Para.p1  -  the emperical risk parameter C 
%      Para.kpar  -  kernel para, include type and para value of kernel
% ______________________________ Output  ______________________________
%     PredictY  -  mt x 1 vector, predicted response variables for TestX 
%     model  -  model related info: alpha, b, nSV, time, etc.
% 
% Written by Lingwei Huang, lateset update: 2021.09.15. 

%% Input 
	C = Para.p1;      ktype = Para.kpar.ktype;      kp1 = Para.kpar.kp1; 
    X = Trn.X;          Y = Trn.Y;         
    clear Trn
    
   if ktype=="lin" % ______________ u'*v
        lib_opt =  sprintf('-t 0 -c %f -b 1 -q', C);
   elseif ktype=="poly" % ________ (gamma*u'*v + coef0)^degree
        gamma = 1;      
        coef0 = 1; 
        degree = kp1; 
        lib_opt =  sprintf('-t 1 -c %f -g %f -r %f -d %f -b 1 -q', C, gamma, coef0, degree);
    elseif ktype=="rbf" % _________ exp(-gamma*|u-v|^2)
        gamma = kp1; 
        lib_opt =  sprintf('-t 2 -c %f -g %f -b 1 -q', C, gamma);
    elseif ktype=="sig" % _________ tanh(gamma*u'*v + coef0)
        gamma = kp1; 
        coef0 = 0; 
        lib_opt =  sprintf('-t 2 -c %f -g %f -r %f -b 1 -h 0 -q', C, gamma, coef0);
	elseif ktype=="pre" % ________ kernel values in training_instance_matrix
        lib_opt =  sprintf('-t 4 -c %f -h 0 -q', C);
   end

%% Training 
	tt = tic;
    
%     kpar = Para.kpar;
%     ValX = KerF(ValX,kpar,X);
%     X = KerF(X,kpar,X);
%     X = sparse(X);
%     lib_opt =  sprintf('-t 0 -c %f -q', C);
    [~, n] = size(X);
    model = LIB_train( Y , X , lib_opt );
    tr_time = toc(tt);
    
%% Predicting
    [mv,~] = size(ValX);       emv = ones(mv,1);
	[PredictY,~,decision_values] = LIB_predict( emv , ValX , model,  '-b 1 -q');  %     decision_values = TestX*model.w + model.b;

%% Output 
     model.w = model.SVs' * model.sv_coef;
     model.b = -model.rho;
      if sum((Y==-1)~=0)
            model.prob = decision_values(:, 1); % probality of samples belonging to class 1
      elseif sum((Y==0)~=0)
            model.prob = decision_values(:, 2); % probality of samples belonging to class 1
       end

    idw0 = ~model.w;
    model.w_ind = ~idw0; % id of useful feature 
    model.spsN = nnz(idw0); % Sparse Number, useless feas # 
    model.nFea = n; 
    model.spsR = model.spsN / n *100; % Sparse Ratio, useless feas % 
    
     model.tr_time = tr_time;
     model.n_SV = model.totalSV;
     model.ind_SV = model.sv_indices;
     
     
     
%      if Para.plt == 1
%          plt.ds = decision_values;
%          plt.ss1 = plt.ds - 1;
%          plt.ss2 = plt.ds + 1;
%          model.plt = plt;
%      end
     
end



% ------------------------------------------------------------------------------------------
% % Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
% % libsvm_options:
% % -s svm_type : set type of SVM (default 0)
% % 	0 -- C-SVC
% % 	1 -- nu-SVC
% % 	2 -- one-class SVM
% % 	3 -- epsilon-SVR
% % 	4 -- nu-SVR
% % -t kernel_type : set type of kernel function (default 2)
% % 	0 -- linear: u'*v
% % 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% % 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% % 	4 -- precomputed kernel (kernel values in training_instance_matrix)
% % -d degree : set degree in kernel function (default 3)
% % -g gamma : set gamma in kernel function (default 1/num_features)
% % -r coef0 : set coef0 in kernel function (default 0)
% % -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% % -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% % -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% % -m cachesize : set cache memory size in MB (default 100)
% % -e epsilon : set tolerance of termination criterion (default 0.001)
% % -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% % -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% % -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% % -v n : n-fold cross validation mode
% % -q : quiet mode (no outputs)

% model.nr_class ---- ���ݼ����ж������
% model.totalSV ---- ֧���������ܸ�����
% model.rho -------- ƫ������෴��(��-b)��
% model.Label ------ ���ݼ������ľ���Ǻţ���Ӧ��nr_class;
% model.sv_indices ----- ֧��������ѵ�����е����������ڼ���ѵ������Ϊ֧����������һ����СΪtotalSV��������;
% model.ProbA&B -- ����������ʹ��-b����ʱ�����õ������ڸ��ʹ���;
% model.nSV -------- ÿ��������֧����������Ŀ��(ע�⣺����nSV�������ǵ�˳����Label��Ӧ)
% model.sv_coef ---- ֧��������Ӧ�Ħ�iyi����һ����СΪtotalSV��������;
% model.SVs -------- ����֧����������ϡ���ʽ�洢����ҪתΪ��ͨ�����ʹ�ú���full;

% % SVM use hyperplanes to perform classification. While performing 
% % classifications using SVM there are 2 types of SVM
% % ��C-SVM
% % ��Nu-SVM
% % C and nu are regularisation parameters which help implement a penalty on 
% % the misclassifications that are performed while separating the classes. 
% % Thus helps in improving the accuracy of the output.
% % C ranges from 0 to infinity and can be a bit hard to estimate and use. 
% % A modification to this was the introduction of nu which operates between 
% % 0-1 and represents the lower and upper bound on the number of examples 
% % that are support vectors and that lie on the wrong side of the hyperplane.
% % Both have a comparative similar classification power, but the nu- SVM 
% % has been harder to optimise.
% % 
% % SVMʹ�ó�ƽ��ִ�з��ࡣʹ��SVMִ�з���ʱ�����������͵�SVM
% % C SVM
% % Nu SVM
% % C��nu�����򻯲����������ڶԷ���ʱִ�еĴ������ʵʩ�ͷ���
% % �����������������׼ȷ�ԡ�
% % C��Χ��0������󣬿����е��ѹ��ƺ�ʹ�á��Դ˵��޸�����0-1֮�����е�nu�����룬
% % ���ұ�ʾ��Ϊ֧�����������ӵ���Ŀ�����޺����ޣ���λ�ڳ�ƽ��Ĵ���ࡣ
% % ���߶��������Ƶķ�����������nu-SVM�����Ż�



% SVMPara =  sprintf('-t 0 -c %f -q',C);
% libmodel = svmtrain(y,X,SVMPara);
% 
% libSVs_idx = libmodel.sv_indices; % libSVs Index
% nlibSVs = length(libSVs_idx); % # of libSVs
% x_SVs = full(libmodel.SVs); % 
% y_SVs = y(libSVs_idx);
% 
% alpha_SVs = libmodel.sv_coef; % actually alpha_i*y_i
% w = sum(diag(alpha_SVs)*x_SVs)'; %����w = sum(alpha_i*y_i*x_i)
% 
% SVs_on_idx = (abs(alpha_SVs)<C); % SV_index on the support hyperplane
% y_SVs_on = y_SVs(SVs_on_idx,:);
% x_SVs_on = x_SVs(SVs_on_idx,:);
% %�����Ͽ�ѡȡ������������߽��ϵ�֧������ͨ�����������ʽ(6.17)���b
% b_temp = zeros(1,sum(SVs_on_idx));%���е�b
% for idx=1:sum(SVs_on_idx)
%     b_temp(idx) = 1/y_SVs_on(idx)-x_SVs_on(idx,:)*w;
% end
% b = mean(b_temp);%��³����������ʹ������֧����������ƽ��ֵ
% 
% %���ֶ��������ƫ����b��svmtrain������ƫ����b�Ա�
% b_model = -libmodel.rho;%model�е�rhoΪ-b


