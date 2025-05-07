function [PredY,model] = LUSI_VSVM_lin(TestX,Data,Para)
%---------- Initiation ----------
gam = Para.p1;%kp
tao =Para.p3;
tao1 = 1-Para.p3;
X = Data.X;
Y = Data.Y;
KerX = X;
KerTstX = TestX;
V = Para.V;
P = Para.P;
%---------- Solve A & b ----------
Ab = inv(KerX'*(tao1*V+tao*P)*KerX+gam*eye(size(X,2)))*KerX'*(tao1*V+tao*P)*Y;
Ac = inv(KerX'*(tao1*V+tao*P)*KerX+gam*eye(size(X,2)))*KerX'*(tao1*V+tao*P)*(ones(length(Y),1));
c = (ones(1,length(Y))*(tao1*V+tao*P)*(Y-KerX*Ab))/(ones(1,length(Y))*(tao1*V+tao*P)*ones(length(Y),1)-ones(1,length(Y))*(tao1*V+tao*P)*KerX*Ac);
A = Ab-c*Ac;
%---------- Output----------
FTrn = (KerX*A)+c;%% Y.Trn output
FTst = (KerTstX*A)+c;%%Y.test output
% FTst = (A'*KerTstX)'+c+Para.modeldv;%%Y.test output
ftst = FTst-0.5;
ftrn = FTrn-0.5;
PredY.tst = sign(ftst);
PredY.trn = sign(ftrn);
model.w = A;
model.b = c;
model.ftrn = ftrn;
model.P = P;
model.trnX = Data.X;
model.prob=FTst;
% if Para.drw == 1
%     drw.ds = f;
%     drw.ss1 = drw.ds - 1;
%     drw.ss2 = drw.ds + 1;
%     model.drw = drw;
%     model.twin = 0;
% end
end
