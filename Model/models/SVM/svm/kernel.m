function K = kernel(X1,X2,kerneltype,can)
% kerneltype �˷���?can ����
% 'linear'�����Ժ�
% 'gaussian'����˹��
% 'mullinear':����ʽ��
switch kerneltype
    case 'linear' % �����ڻ� K(v1,v2) = <v1,v2>
        K = X1'*X2;
    case 'gaussian'% K(v1,v2)=exp(-gama||v1-v2||^2)
        delta = can;
        XX = sum(X1'.*X1',2);
        YY = sum(X2'.*X2',2);
        XY = X1'*X2;
        K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
        if isstruct(delta)
            K = exp(-K./delta.c1);
        else
            K = exp(-K./delta);
        end
% K = exp(-delta * (pdist2(	X1',X2').^2)  );  
    case 'mullinear' % K<v1,v2> = (<v1,v2>+c) d;dΪ����ʽ�Ĵ�����cΪ����ʽ��λ�Ƴ���?
        K = (X1'*X2).^can;
end
end



