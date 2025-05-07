function P = Pcaculate(X,Y,ptype)
Ptype = num2str(ptype);
Y(Y==-1,1)=0;%%
switch  Ptype
    case '1'
        P =diag(Y)*diag(Y)';
    case '2'
        P =eye(length(Y))*(eye(length(Y)))';
    case '3'
        P =X*X';
    case '4'
        r_p = X*pca(X);%ͨ��ĳ������ͶӰ������ά������ӳ�䵽��ά�Ŀռ��У�����������ͶӰ��ά�������ݵ���Ϣ����󣨷�����󣩣��Դ�ʹ�ý��ٵ�����ά�ȣ�ͬʱ����ס�϶��ԭ���ݵ�����ԡ�
        P = r_p(:,1)*r_p(:,1)';
    case  '5'
        [P]=L1(X,Y);
    case  '6'
        [P]=L2(X,Y);
    case  '7'
        [P]=L3(X,Y);
    case'8'
        D1 = X(:,1);
        P = D1*D1';
    case'9'
        P =   ones(length(Y),1)*ones(length(Y),1)';
    case'10'
        P =   Y*Y';
end
    end