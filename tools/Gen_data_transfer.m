clear
clc
rng(1);
datasave=["on"];
% datasave=["off"];
f=["on"];
% f=["off"];
num=100;
Gentype1_A1='normal'; p1_1_A1=18 ; p2_1_A1=3; 
Gentype0_A1='normal'; p1_0_A1=22 ; p2_0_A1=3;

Gentype1_A2='normal'; p1_1_A2=18 ; p2_1_A2=3; 
Gentype0_A2='normal'; p1_0_A2=22 ; p2_0_A2=3;

Gentype1_A3='normal'; p1_1_A3=-3 ; p2_1_A3=3; 
Gentype0_A3='normal'; p1_0_A3=-7 ; p2_0_A3=3;

Gentype1_A4='normal'; p1_1_A4=-3 ; p2_1_A4=3; 
Gentype0_A4='normal'; p1_0_A4=-7 ; p2_0_A4=3;

%% >>>>>>>>>>>>>>>>>>>> Source Domain 1 >>>>>>>>>>>>>>>>>>>>
%-------------------- Distribution Para --------------------
X_P_A1 = normrnd(p1_1_A1,p2_1_A1,num, 2);
X_P_A1(:,1)=X_P_A1(:,1)-22;
X_N_A1 = normrnd(p1_0_A1,p2_0_A1,num, 2);
X_N_A1(:,1)=X_N_A1(:,1)-22;
X_A1=[X_P_A1;X_N_A1];
%-------------------- Positive and negative probability density --------------------
Xppdf_A1 = pdf(Gentype1_A1, X_P_A1, p1_1_A1, p2_1_A1);
Xnpdf_A1 = pdf(Gentype0_A1, X_N_A1, p1_0_A1, p2_0_A1);
%-------------------- Label Y --------------------
Y_A1=zeros(length(X_A1),1);
for i = 1:length(X_A1)
    if i<=length(X_P_A1)
        Y_A1(i)=1;
    else 
        Y_A1(i)=0;
    end
end
%-------------------- train and test --------------------
X_train_A1=X_A1; Y_train_A1=Y_A1;

%% >>>>>>>>>>>>>>>>>>>> Source Domain 2 >>>>>>>>>>>>>>>>>>>>
%-------------------- Distribution Para --------------------
X_P_A2 = normrnd(p1_1_A2,p2_1_A2,num, 2);
X_N_A2 = normrnd(p1_0_A2,p2_0_A2,num, 2);
X_A2=[X_P_A2;X_N_A2];
%-------------------- Positive and negative probability density --------------------
Xppdf_A2 = pdf(Gentype1_A2, X_P_A2, p1_1_A2, p2_1_A2);
Xnpdf_A2 = pdf(Gentype0_A2, X_N_A2, p1_0_A2, p2_0_A2);
%-------------------- Label Y --------------------
Y_A2=zeros(length(X_A2),1);
for i = 1:length(X_A2)
    if i<=length(X_P_A2)
        Y_A2(i)=1;
    else 
        Y_A2(i)=0;
    end
end
%-------------------- train and test --------------------
X_train_A2=X_A2; Y_train_A2=Y_A2;

%% >>>>>>>>>>>>>>>>>>>> Source Domain 3 >>>>>>>>>>>>>>>>>>>>
%-------------------- Distribution Para --------------------
X_P_A3 = normrnd(p1_1_A3,p2_1_A3,num, 2);
X_N_A3 = normrnd(p1_0_A3,p2_0_A3,num, 2);
X_A3=[X_P_A3;X_N_A3];
%-------------------- Positive and negative probability density --------------------
Xppdf_A3 = pdf(Gentype1_A3, X_P_A3, p1_1_A3, p2_1_A3);
Xnpdf_A3 = pdf(Gentype0_A3, X_N_A3, p1_0_A3, p2_0_A3);
%-------------------- Label Y --------------------
Y_A3=zeros(length(X_A3),1);
for i = 1:length(X_A3)
    if i<=length(X_P_A3)
        Y_A3(i)=1;
    else 
        Y_A3(i)=0;
    end
end
%-------------------- train and test --------------------
X_train_A3=X_A3; Y_train_A3=Y_A3;

%% >>>>>>>>>>>>>>>>>>> Source Domain 4 >>>>>>>>>>>>>>>>>>>>
%-------------------- Distribution Para --------------------
X_P_A4 = normrnd(p1_1_A4,p2_1_A4,num, 2);
X_P_A4(:,1)=X_P_A4(:,1)+22;
X_N_A4 = normrnd(p1_0_A4,p2_0_A4,num, 2);
X_N_A4(:,1)=X_N_A4(:,1)+22;
X_A4=[X_P_A4;X_N_A4];
%-------------------- Positive and negative probability density --------------------
Xppdf_A4 = pdf(Gentype1_A4, X_P_A4, p1_1_A4, p2_1_A4);
Xnpdf_A4 = pdf(Gentype0_A4, X_N_A4, p1_0_A4, p2_0_A4);
%-------------------- Label Y --------------------
Y_A4=zeros(length(X_A4),1);
for i = 1:length(X_A4)
    if i<=length(X_P_A4)
        Y_A4(i)=1;
    else 
        Y_A4(i)=0;
    end
end
%--------------------  train and test --------------------
X_train_A4=X_A4; Y_train_A4=Y_A4;
%------------------------- Figure -------------------------
if sum(f=='on')
    % Sample Points
    figure
    yticks(0:10:100);

    plot(X_P_A4(:,1), X_P_A4(:,2), '.','MarkerSize',10,'Color',[000/255,000/255,000/255])
    hold on
    
    plot(X_N_A4(:,1), X_N_A4(:,2), 'o','MarkerSize',5,'Color',[000/255,000/255,000/255])
    hold on

    plot(X_P_A1(:,1), X_P_A1(:,2), '.','MarkerSize',12,'Color',[022/255,006/255,138/255])
    hold on

    plot(X_P_A2(:,1), X_P_A2(:,2), '.','MarkerSize',12,'Color',[068/255,117/255,122/255])
    hold on

    plot(X_P_A3(:,1), X_P_A3(:,2), '.','MarkerSize',12,'Color',[144/255,190/255,224/255])
    hold on

    plot(X_P_A4(:,1), X_P_A4(:,2), '.','MarkerSize',12,'Color',[191/255,030/255,046/255])
    hold on
    
    plot(X_N_A1(:,1), X_N_A1(:,2), 'o','MarkerSize',5,'Color',[022/255,006/255,138/255])
    hold on

    plot(X_N_A2(:,1), X_N_A2(:,2), 'o','MarkerSize',5,'Color',[068/255,117/255,122/255])
    hold on

    plot(X_N_A3(:,1), X_N_A3(:,2), 'o','MarkerSize',5,'Color',[144/255,190/255,224/255])
    hold on

    plot(X_N_A4(:,1), X_N_A4(:,2), 'o','MarkerSize',5,'Color',[191/255,030/255,046/255])
%     tt=['Para PS=',num2str(p1_1),'&',num2str(p2_1), ' | NS= ',num2str(p1_0),'&',num2str(p2_0),' || Para PT=',num2str(p1_1_T),'&',num2str(p2_1_T), ' | NT= ',num2str(p1_0_T),'&',num2str(p2_0_T),];
%     title(tt)
    xlim([-20,45])
    ylim([-20,35])
    legend('Class 1','Class 0','Domain 1', 'Domain 2', 'Domain 3', 'Domain 4',"Location","best" ,'FontWeight', 'bold')
    grid on
    grid minor
    box off
    xlabel("X1")
    ylabel("X2")
    set(gca, 'FontWeight', 'bold');  % 设置坐标轴文本加粗
    set(get(gca, 'xlabel'), 'FontWeight', 'bold'); % 设置 x 轴标签加粗
    set(get(gca, 'ylabel'), 'FontWeight', 'bold'); % 设置 y 轴标签加粗
end
%-------------------- Auto save --------------------
if sum(datasave=='on')
    g_T=Gentype1_A4;
    h_T=Gentype0_A4;
    v_T=strcat(g_T,h_T);
    a_T=[v_T,num2str(i),'.mat'];
    save(a_T,'X_train_A1','Y_train_A1','X_train_A2','Y_train_A2','X_train_A3','Y_train_A3','X_train_A4','Y_train_A4')
    saveas(gcf, [v_T,num2str(i),'.png']);
    saveas(gcf, [v_T,num2str(i),'.eps']);
end

