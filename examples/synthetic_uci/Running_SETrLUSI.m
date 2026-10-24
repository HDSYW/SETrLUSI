clear;clc;close all;warning off
pa.DA=["rice123"];
R=(1:10);
Para.time=datestr(now, 30);
for NNN=1:numel(pa.DA)
    Name=pa.DA(NNN);
    diary 'Diary.txt';
    Para. AutoRec = "ON";
    % Para. AutoRec = "OFF";
    % pa.figure="On";
    pa.figure="Off";
    %     Figure="On";
    Figure="Off";
    % Synthetic="ON";
    Synthetic="OFF";
    %% ============================== Ⅰ Running Setting ==============================
    % ----------◆ -1- Model Selecting ◆----------
    ModS = [];
    ModS = [ModS;"SETrLUSI_VSVM_Ram_New"];
%    ModS = [ModS;"MSDTradaboost"];
%    ModS = [ModS;"MutiTradaboost"];
%    ModS = [ModS;"ThreeSW_MSTL"];
%    ModS = [ModS;"METL"];
%    ModS = [ModS;"MHTLAdaBoost"];
%    ModS = [ModS;"WMSTradaboost"];
%    ModS = [ModS;"Tradaboost"];
    % ----------◆ -2- Feature Kernel types ◆----------
    Para.kpar.ktype = 'lin'; % poly or lin or rbf;
    % ----------◆ -3- V & P  Kernel types ◆----------
    P=["2"]; V_Matrix = 'Vmatrix'  ; V_MatrixFun = str2func(V_Matrix) ; Para.vmatrix = V_Matrix;
    Para.CDFx = 'uniform' ; Para.v_ker = 'theta_ker'      ; CDFx = Para.CDFx;
    % ----------◆ -4- Files ◆----------
    name= Name; Para.name=name;
    % ----------◆ -5- Repeat ◆----------
    Repeat=numel(R);
    % ----------◆ -6- K-Fold ◆----------
    k=3;
    % ----------◆ -7- Para Range ◆----------
    pa.min = -8  ;  pa.step =  2 ;  pa.max = 8;
    pa.taomin=0.1 ; pa.taomax=0.9 ; pa.taostep=0.4 ;
    pa.T=100;
    % ----------◆ -8- Testproportion ◆----------
    Testproportion_A=0.9 ; Testproportion_T=0.9;
    %% ============================== Ⅱ Running Procedure ==============================
    for ms = 1 : length(ModS)
        Mod = ModS(ms); res.Mod=Mod;tic;
        for chongfu=1:Repeat
            randomindex=randperm(1000,100); seed=randomindex(chongfu); res.seed=R(chongfu); rng(R(chongfu)); res.chongfu=chongfu;
            for l = 1:length(name)
                f = 'Data/';  G = name(l) ;  folder = f+G;  files = dir(fullfile(folder, '*.mat'));
                for p = 1:length(files)
                    filename = fullfile(folder, files(p).name) ; data_ori = load(filename) ; SVMFun = str2func(Mod);
                    fprintf('%s\n', repmat('=', 1, 60)); fprintf('Proc===>%s\t\n',num2str(chongfu));fprintf('File===>%s\t\n',G);
                    fprintf('Seed===>%s\t\n',num2str(res.seed)); fprintf('Mod===>%s\t\n',Mod);
                    % ----------◆ -1- Split Train and Test ◆----------
                    [X_train_T,Y_train_T,X_test_T,Y_test_T] = TT(data_ori.X_test,data_ori.Y_test ,Testproportion_T);X_train=[];Y_train=[];
                    if iscell(data_ori.X_train_A)
                        for i = 1: numel(data_ori.X_train_A)
                            X_train=[X_train;data_ori.X_train_A{i}]; Y_train=[Y_train;data_ori.Y_train_A{i}];
                        end
                    else
                        X_train=[data_ori.X_train_A;X_train_T]; Y_train=[data_ori.Y_train_A;Y_train_T];
                    end
                    if size(X_train,2)~=size(X_train_T,2)
                        [COEFF, SCORE] =pca(X_train);
                        X_train=SCORE(:,1:size(X_train_T,2));
                        data_ori.X_train_A{1}=X_train(1:size(data_ori.X_train_A{1},1),:);
                        data_ori.X_train_A{2}=X_train(size(data_ori.X_train_A{1},1)+1:end,:);
                    end
                    % ----------◆ -2- Different Models ◆----------
                    if sum(Mod=='SETrLUSI_VSVM_Ram_New')
                        pa.Vtype=["I"] ;
                        % pa.Vtype=["V"] ;
                        pa.ram=["All Open"];
                        pa.p3=0.5; pa.H=pa.T;
                        pa.randomratio=0.9; % the ratio of Ds
                        pa.predicate_ori=["one","Zero","DV","A_Y","feature","Kernel","X","XX","Ram_X","Ram_XX"];
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = SETrLUSI_VSVM_Ram_New(Data,Para.kpar.ktype,Para.v_ker,CDFx,pa) ;  result=catstruct(res,Res);TELUSI_test_error=Res.testerror;
                    end
                    if sum(Mod=='Tradaboost')
                        Data.X_train_A=X_train ; Data.Y_train_A=Y_train;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = Tradaboost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);tradsvm_test_error=Res.testerror;
                    end
                    if sum(Mod=='MutiTradaboost')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = MutiTradaboost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);mutitradsvm_test_error=Res.testerror;
                    end
                    if sum(Mod=='WMSTradaboost')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = WMSTradaboost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);WMSTradsvm_test_error=Res.testerror;
                    end
                    if sum(Mod=='TaskTradaboost')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = TaskTradaboost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);mutitradsvm_test_error=Res.testerror;
                    end
                    if sum(Mod=='MSDTradaboost')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = MSDTradaboost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);MSDTradaboost_test_error=Res.testerror;
                    end
                    if sum(Mod=='MHTLAdaBoost')
                        pa.k=2;
                        Data.X_train_A=X_train ; Data.Y_train_A=Y_train;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = MHTLAdaBoost_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);MHTLAdaBoost_test_error=Res.testerror;
                    end
                    if sum(Mod=='METL')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = METL_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);
                    end
                    if sum(Mod=='ThreeSW_MSTL')
                        Data.X_train_A=data_ori.X_train_A ; Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T                ; Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T                   ; Data.Y_test_T=Y_test_T;
                        Res = ThreeSW_MSTL_F(Data,Para.kpar.ktype,pa) ; result=catstruct(res,Res);
                    end
                end
            end
            %% ============================== Ⅲ Result Display ==============================
            Para.runtime=toc;
            ResultsInfo(result,Para)
            MAC(chongfu)=result.ac_test ; MF(chongfu)=result.F ; MGM(chongfu)=result.GM ;
            MSpe(chongfu)=result.Spe ; MSen(chongfu)=result.Sen;
        end
        fprintf('AC=%s\t',num2str(sprintf('%.2f', mean(MAC))))
        fprintf('AC_Std=%s\t\n',num2str(sprintf('%.2f', std(MAC))))
        fprintf('FM=%s\t',num2str(sprintf('%.2f', mean(MF))))
        fprintf('F_Std=%s\t\n',num2str(sprintf('%.2f', std(MF))))
        fprintf('GM=%s\t',num2str(sprintf('%.2f', mean(MGM))))
        fprintf('GM_Std=%s\t\n',num2str(sprintf('%.2f', std(MGM))))
        fprintf('Spe=%s\t',num2str(sprintf('%.2f', mean(MSpe))))
        fprintf('Spe_Std=%s\t\n',num2str(sprintf('%.2f', std(MSpe))))
        fprintf('Sen=%s\t',num2str(sprintf('%.2f', mean(MSen))))
        fprintf('Sen_Std=%s\t\n',num2str(sprintf('%.2f', std(MSen))))
        fprintf('%s\n', repmat('=', 1, 60))
        clear result
    end
    diary off
end