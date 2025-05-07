clear
clc
rng(1);
Synthetic=["on"];
% Synthetic=["off"];
% UCI=["on"];
UCI=["off"];
% DUCI=["on"];
DUCI=["off"];
% datasave=["on"];
datasave=["off"];
f=["on"];
% f=["off"];
% PCA=["on"];
PCA=["off"];
% load("BankPersonal.mat");
%% >>>>>>>>>>>>>>>>>>>> Synthetic Data >>>>>>>>>>>>>>>>>>>>
if sum(Synthetic=='on')
%-------------------- Generate X ------------ --------
% X_P=sort(rand(1,250)*-30+5,'ascend');
% X_N=sort(rand(1,250)*30-5,'ascend');
% X= [X_P,X_N]';
%-------------------- Distribution Para --------------------
name='ddt2';
Gentype1='normal'; p1_1=1 ; p2_1=2; t1=1;
Gentype2='normal'; p1_2=5 ; p2_2=6; t2=0;
% Gentype1_testp='normal'; p1_tp_1=-10 ; p1_tp_2=5; 
% Gentype1_testn='normal'; p1_tn_1=10 ; p1_tn_1=5; 
Gentype0='normal'; p1_0=4 ; p2_0=3;
X_P = normrnd(p1_1,p2_1,150, 1);
X_N = normrnd(p1_0,p2_0,150, 1);
% X_P = chi2rnd(p1_1, 400, 1);
% X_N = chi2rnd(p1_0,  80, 1); 
X_P=sort(X_P,'ascend');
X_N=sort(X_N,'ascend');
%------对称------
% for k=1:size(X_P)
%         X_N(k)=12+(12-X_P(k));
% end
% X_N=X_N';
X=[X_P;X_N];
%-------------------- Positive and negative probability density --------------------
Xppdf = t1*pdf(Gentype1, X_P, p1_1, p2_1)+t2*pdf(Gentype2, X_P, p1_2, p2_2);
% Xnpdf=Xppdf;
Xnpdf = pdf(Gentype0, X_N, p1_0, p2_0);
%-------------------- Label Y --------------------
Y=zeros(length(X),1);
for i = 1:length(X)
    if any(X_P==X(i))
        Y(i)=1;
    else 
        Y(i)=2;
    end
end
%-------------------- Class probability --------------------
pyp=length(find(Y(Y==1)))/length(Y); %probability of Y=1
pyn=1-pyp;                           %probability of Y=-1
%-------------------- Divide the train and test --------------------
Data=[X,Y];
for i =1:10
    X_train=[];  Y_train=[];  X_test=[]; Y_test=[];
    scale=0.1; % training set scale
    for label=1:length(unique(Y))
        idx = find(Y==label);
        num = int32(length(idx)*scale);
        train = idx(randperm(length(idx),num));  
        test = setdiff(idx,train);   
        X_train = [X_train;Data(train,1:end-1)];
        Y_train = [Y_train;Y(train)];
        X_test = [X_test;Data(test,1:end-1)];
        Y_test = [Y_test;Y(test)];
    end
%-------------------- Put out --------------------
    Xppdf_test = pdf(Gentype1, X_test, p1_1, p2_1);
    Xnpdf_test = pdf(Gentype0, X_test, p1_0, p2_0);
    m1 = size(X_P, 1);
    m2 = size(X_N, 1);
    indte1 = 1:m1;
    indte2 = 1:m2;
    Y_train(Y_train==2)=-1;
    Y_test(Y_test==2)=-1;
%-------------------- Auto save --------------------
    if sum(datasave=='on')
        g=Gentype1;
        h=Gentype0;
        v=strcat(g,h);
        a=[v,num2str(i),'.mat'];
        save(a,'X_train','X_test','Y_train','Y_test','Xppdf_test','Xnpdf_test','pyp','pyn','indte1','indte2')
    end
end
end
%------------------------- Figure -------------------------
if sum(f=='on')
    figure
    ValXP = X_P;
    ValXN = X_N;
    m1 = size(X_P, 1);
    m2 = size(X_N, 1);
    indte1 = 1:m1;
    indte2 = 1:m2;
    % PDF
    plot(X_P, Xppdf(indte1, :), 'r^-')
    hold on
    plot(X_N, Xnpdf(indte2, :), 'bs-')
    hold on
    % Sample Points
%     plot(ValXP, zeros(size(ValXP)), 'r.','MarkerSize',20)
%     hold on
%     plot(ValXN, zeros(size(ValXN))+0.01, 'bo')
%     xlim([-25 25])
%     ylim([0 0.1])
    legend('Pos PDF', 'Neg PDF')
    ylim([0,0.2])
    saveas(gcf, [name,'.png']);
    % CDF
    figure
    allprob_test =  Xppdf_test*pyp + Xnpdf_test*(1-pyn);
    postprob_test = (Xppdf_test*pyp)./allprob_test;
    dd=[X_test,Y_test,postprob_test];
    dd=sortrows(dd,1);
    X_test=dd(:,1);
    Y_test=dd(:,2);
    postprob_test=dd(:,3);
    indP_test = find(Y_test==1);
    indN_test = find(Y_test==-1);
    m = size(X_test, 1);
    indte = 1:m;
    ValXP = X_test(indP_test);
    ValYP = Y_test(indP_test);
    ValXN = X_test(indN_test);
    ValYN = Y_test(indN_test);
    
    plot(ValXP, zeros(size(ValXP)), 'r<')
    hold on
    plot(ValXN, zeros(size(ValXN)), 'b>')
    plot(X_test, postprob_test(indte, :), 'ko-')
end
%------------------------- multi D normal -------------------------
% mu = [5 8];
% Sigma = [8 5; 5 8];
% X = mvnrnd(mu,Sigma,100);
% rng('default')  % For reproducibility
% y = mvnpdf(X,mu,Sigma);
% mu_n = [-5 8];
% Sigma_n = [8 5; 5 8];
% X_n = mvnrnd(mu_n,Sigma_n,100);
% y_n = mvnpdf(X_n,mu_n,Sigma_n);
% 
% figure
% scatter3(X(:,1),X(:,2),y,'blue')
% hold on
% scatter3(X_n(:,1),X_n(:,2),y_n,'red')
%% >>>>>>>>>>>>>>>>>>>> UCI Data Preprocessing >>>>>>>>>>>>>>>>>>>>
if sum(UCI=='on')
%     Y(Y==-1)=0;
%     data=[X,Y];
%     data_z=data(data(:,end)==1,:);
%     data_f=data(data(:,end)==0,:);
%     randomindices_1 = randperm(size(data_z,1),floor(0.2*size(data_z,1)));
%     randompoints_1 = data_z(randomindices_1,:);
%     randomindices_0 = randperm(size(data_f,1),floor(0.2*size(data_f,1)));
%     randompoints_0 = data_f(randomindices_0,:);
%     data_01=[randompoints_1;randompoints_0];
%     X=DATA(:,1:end-1);
%     Y=DATA(:,end);
%     % DATA=table2array(breast);
    % ------------------------- Standard -------------------------
%     X=normalize(X,'zscore');
    data=[X,Y];
    data = data(~any(isnan(data), 2), :);
    X=data(:,1:end-1);
    Y=data(:,end);
%     X=mapminmax(X',0,1)';
    % ------------------------- PCA -------------------------
    if sum(PCA=='on')
        [coeff,score,latent,tsquared,explained,mu] = pca(X,'Centered',false);
        s=0;i=0;
            while i<90 %解释性90
                s=s+1;
                i=i+explained(s);
            end
        DATA=[score(:,1:s),Y];
    else
%         DATA=[X,Y];
    end
    % ---------------------------------------------------------
%     DATA=[X(:,[1:3]),Y];
    
    for i = 1%:10
    %     [m,n]= find(isnan(DATA));
%         DATA(m,:)=[];
    %     DATA=unique(DATA,'rows');
        id=find(DATA(:,end)==2);
        DATA(id,end)=0;
        id=find(DATA(:,end)==-1);
        DATA(id,end)=0;
        data_z=DATA((DATA(:,end)==1),:);
        data_f=DATA((DATA(:,end)==0),:);
        % ------------------------- Test -------------------------
%         idx_z=randperm(size(data_z,1), floor(0.99*size(data_z,1)))';
%         idx_f=randperm(size(data_f,1), floor(0.99*size(data_f,1)))';
        idx_z=randperm(size(data_z,1),size(data_z,1)-50)';
%         idx_f = randsample(size(data_f,1), 2*size(data_z,1));

        idx_f=randperm(size(data_f,1), size(data_f,1)-50)';
        D_test=[(data_z(idx_z,:));(data_f(idx_f,:))];
        X_test=D_test(:,1:end-1);
        Y_test=D_test(:,end);
        % ------------------------- Train -------------------------
        idx1_z=setdiff((1:size(data_z,1))',idx_z,'row','stable');
        idx1_f=setdiff((1:size(data_f,1))',idx_f,'row','stable');
        idx1_z2=randsample(size(idx1_z,1),50);
        idx1_f2=randsample(size(idx1_f,1),50);
        D_train = [(data_z(idx1_z2,:));(data_f(idx1_f2,:))];
        X_train=D_train(:,1:end-1);
        Y_train=D_train(:,end);
        Y_train(Y_train==0)=-1;
        Y_test(Y_test==0)=-1;
        % ------------------------- Auto save -------------------------
        if sum(datasave=='on')
            a=['BankPersonal',num2str(i),'.mat'];
            save(a,'X_train','X_test','Y_train','Y_test')
        end
    end
end
%% >>>>>>>>>>>>>>>>>>>> DUCI Data Preprocessing >>>>>>>>>>>>>>>>>>>>
if sum(DUCI=='on')
%     Y(Y==-1)=0;
%     data=[X,Y];
%     data_z=data(data(:,end)==1,:);
%     data_f=data(data(:,end)==0,:);
%     randomindices_1 = randperm(size(data_z,1),floor(0.2*size(data_z,1)));
%     randompoints_1 = data_z(randomindices_1,:);
%     randomindices_0 = randperm(size(data_f,1),floor(0.2*size(data_f,1)));
%     randompoints_0 = data_f(randomindices_0,:);
%     data_01=[randompoints_1;randompoints_0];
%     X=DATA(:,1:end-1);
%     Y=DATA(:,end);
    % DATA=table2array(breast);
    % ------------------------- Standard -------------------------
    X=normalize(X,'zscore');
    % ------------------------- PCA -------------------------
%     [coeff,score,latent,tsquared,explained,mu] = pca(X,'Centered',false);
%     DATA=[score(:,1:2),Y];
%     DATA=data;
    DATA=[X,Y];
    for i = 1%:10
    %     [m,n]= find(isnan(DATA));
%         DATA(m,:)=[];
    %     DATA=unique(DATA,'rows');
        id=find(DATA(:,end)==2);
        DATA(id,end)=0;
        id=find(DATA(:,end)==-1);
        DATA(id,end)=0;
        data_z=DATA((DATA(:,end)==1),:);
        data_f=DATA((DATA(:,end)==0),:);
        % ------------------------- Test -------------------------
        idx_z=randperm(size(data_z,1), floor(0.4*size(data_z,1)))';
        idx_f=randperm(size(data_f,1), floor(0.4*size(data_f,1)))';
        
        D_test=[(data_z(idx_z,:));(data_f(idx_f,:))];
        X_test=D_test(:,1:end-1);
        Y_test=D_test(:,end);
        % ------------------------- Train -------------------------
        a=(1:size(data_z,1))';
        b=(1:size(data_f,1))';
        idx1_z=setdiff(a,idx_z,'row','stable');
        idx1_f=setdiff(b,idx_f,'row','stable');
        
        D_train = [(data_z(idx1_z,:));(data_f(idx1_f,:))];
        randomindices=randperm(size(D_train,1),floor(0.96*size(D_train,1)))';
        X_train_A=D_train(randomindices,1:end-1);
        X_train=D_train(setdiff((1:size(D_train,1))',randomindices,'row','stable'),1:end-1);
        Y_train_A=D_train(randomindices,end);
        Y_train=D_train(setdiff((1:size(D_train,1))',randomindices,'row','stable'),end);
        Y_train(Y_train==0)=-1;
        Y_test(Y_test==0)=-1;
        % ------------------------- Auto save -------------------------
        if sum(datasave=='on')
            a=['dvdvsbook',num2str(i),'.mat'];
            save(a,'X_train_A','X_train','X_test','Y_train_A','Y_train','Y_test')
        end
    end
end
% load('caltech vs amazon')
% perc=0.05;
% Y_test(Y_test==2)=-1;
% Y_train_A(Y_train_A==2)=-1;
% test=[X_test,Y_test];
% %  train=[X_train,Y_train];
%  idxz=test(:,end)==1;
%  zz=test(idxz,:);
%  idxf=test(:,end)==-1;
%  ff=test(idxf,:);
%  X_train=[zz(1:floor(perc*size(zz,1)),1:end-1);ff(1:floor(perc*size(ff,1)),1:end-1)];
%  X_test =[zz((floor(perc*size(zz,1))+1):end,1:end-1);ff((floor(perc*size(zz,1))+1):end,1:end-1)];
%  Y_train=[zz(1:floor(perc*size(zz,1)),end);ff(1:floor(perc*size(ff,1)),end)];
%  Y_test =[zz((floor(perc*size(zz,1))+1):end,end);ff((floor(perc*size(ff,1))+1):end,end)];
% figure
%  close,figure,hold on,
% plot(X(Y_train==1,1),X(Y_train==1,2),'o')
% plot(X(Y_train==-1,1),X(Y_train==-1,2),'o')
