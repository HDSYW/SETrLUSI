function [P,tao,phi] = DPcaculate(X, T_Y, A_X, P_Y, A_Y, DV, KP, ptype, paratao)
Ptype = num2str(ptype);
tao=paratao;
P_Y(P_Y==-1)=0;%%
A_Y(A_Y==-1)=0;%%
T_Y(T_Y==-1)=0;%%
switch  Ptype
    case 'A_Y'
        P = P_Y*P_Y';
        phi=P_Y;
    case 'DV'
        P = DV*DV';
        phi=DV;
    case 'Zero'
        P=zeros(size(X,1),size(X,1));
        tao=0;
        phi=zeros(size(X,1),1);
    case 'Kernel'
        K=KerF(X,KP,A_X);
        P_=zeros(size(K,1),size(K,1));
        for i = 1:size(K,2)
            P_=P_+K(:,i)*K(:,i)';
        end
        P=P_./size(K,2);
        phi=K;
    case  'feature'
        A_Y(A_Y==0)=-1;
        model = LIB_train(A_Y , A_X , '-t 0 -c 1 -q');
        w = model.SVs' * model.sv_coef;
        ind=find(max(abs(w')));
        P=X(:,ind)*X(:,ind)';
        phi=X(:,ind);
    case  'Spearman1'
        relation=corr(A_X,A_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)';
        phi=X(:,ind);
    case  'Kendall1'
        relation=corr(A_X,A_Y,'type',"Kendall");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)';
        phi=X(:,ind);
    case  'pearson1'
        relation=corr(A_X,A_Y,'type',"pearson");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)';
        phi=X(:,ind);
    case  'Spearman2'
        relation=corr(A_X,A_Y,'type',"Spearman");
        [secondorder,secondidx]=sort(relation,"descend");
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)';
        phi=X(:,idx);
    case  'Kendall2'
        relation=corr(A_X,A_Y,'type',"Kendall");
        [secondorder,secondidx]=sort(relation,"descend");
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)';
        phi=X(:,idx);
    case  'pearson2'
        relation=corr(A_X,A_Y,'type',"pearson");
        [secondorder,secondidx]=sort(relation,"descend");
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)';
        phi=X(:,idx);
    case  'Spearman_T'
        relation=corr(X,T_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)';
        phi=X(:,ind);
    case  'T_Y'
        P=T_Y*T_Y';
    case 'X'
        P_X=zeros(size(X,1),size(X,1));
        for i = 1:size(X,2)
            P_X=P_X+X(:,i)*X(:,i)';
        end
        P=P_X./size(X,2);
    case 'Ram_X'
        i = randi(size(X,2));
        P=X(:,i)*X(:,i)';
%     case 'A_X'
%         P_X=zeros(size(A_X,1),size(A_X,1));
%         for i = 1:size(A_X,2)
%             P_X=P_X+A_X(:,i)*A_X(:,i)';
%         end
%         P=P_X./size(A_X,2);
%     case 'Ram_A_X'
%         i = randi(size(A_X,2));
%         P=A_X(:,i)*A_X(:,i)';
%     case 'A_XA_X'
%         A_XA_X=A_X*A_X';
%         P_X=zeros(size(A_XA_X,1),size(A_XA_X,1));
%         for i = 1:size(A_XA_X,2)
%             P_X=P_X+A_XA_X(:,i)*A_XA_X(:,i)';
%         end
%         P=P_X./size(A_X,2);
%     case 'Ram_A_XA_X'
%         A_XA_X=A_X*A_X';
%         i = randi(size(A_XA_X,2));
%         j = randi(size(A_XA_X,2));
%         P=A_XA_X(:,i)*A_XA_X(:,j)';
    case 'XX'
        XX=X*X';
        P_XX=zeros(size(XX,1),size(XX,1));
        for i = 1:size(XX,2)
            for j=i:size(XX,2)
                P_XX=P_XX+XX(:,i).*XX(:,j);
            end
        end
        P=P_XX./size(XX,2);
    case 'Ram_XX'
        XX=X*X';
        i = randi(size(XX,2));
        j=randi(size(XX,2));
        P=XX(:,i)*XX(:,j)';
    case 'one'
        P=ones(size(X,1),1)*ones(size(X,1),1)';
end
end