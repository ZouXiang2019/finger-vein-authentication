clc;
clear;
%声音幅值
a="大拇指";   s="食指";   d="中指";   f="无名指";   g="小拇指";
whaleFile = 'E:\blood flow of finger vein\data\row_data\4k18kchirp\右'+g+'.m4a';
q="thumb";   w="indexfinger";   e="middlefinger";   r="ringfinger";   t="littlefinger";   
[x,fs] = audioread(whaleFile);
% y=bandp(x,3900,18100,3800,18200,0.1,30,fs);%带通滤波
y=highp(x,3900,3800,0.1,30,fs);%高通
% zou_fft(y(:,1),fs)
% figure;plot(abs(y(:,1)));
% xlabel('Sample Number')
% ylabel('Amplitude')
% % 音频剪切
start_time = 2;
end_time = 10;
% X_new=left((fs*start_time+1):fs*end_time,1);
% Y_new=right((fs*start_time+1):fs*end_time,1);
feature=y((fs*start_time+1):fs*end_time,:);
% feature = [X_new,Y_new];
%保存.mat文件
save('E:\blood flow of finger vein\data\给建伟的\4k18kchirp+highpass\right_'+t+'.mat','feature') 
%每50个采样点求均值
% i=length(X_new)/50;
% for j=1:i
%     RSS(j)=mean(X_new((j-1)*50+1:j*50));
%     phase(j)=mean(Y_new((j-1)*50+1:j*50));
% end
%生成特征矩阵% 7680*2*50
% i=length(X_new)/50;
% for j=1:i
% %     for k=1:1
%         RSS(j,1:50)=X_new(((j-1)*50+1):j*50).';
%         phase(j,1:50)=Y_new((j-1)*50+1:j*50).';
%         feature{j,1}=[RSS(j,1:50);phase(j,1:50)];
% %     end
% end
%生成csv
% EPC = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,31,32,33,34,35,36,37,41,42,43,44,45,46,47,51,52,53,54,55,56,57,61,62,63,64,65,66,67,71,72,73,74,75,76,77];
% for i=1:49
%     csv((i-1)*5320+1:5320*i,1)="E0"+EPC(i);
%     csv((i-1)*5320+1:5320*i,3)=X_new((i-1)*5320+1:5320*i);
%     csv((i-1)*5320+1:5320*i,4)=Y_new((i-1)*5320+1:5320*i);
% end
% csv(1:260680,2)=1;%time
% 
% %生成csv文件
% % name=file_name;%图片名称
% % m1=m';%转置
% various={'EPC','Timestamp','RSS','PhaseAngle'};%表头
% result_table=table(csv(:,1),csv(:,2),csv(:,3),csv(:,4),'VariableNames',various);%表的内容
% writetable(result_table, '左中指.csv')%创建csv表格

