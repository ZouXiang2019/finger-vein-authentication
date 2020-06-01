%声音幅值
whaleFile = 'air1+finger1.m4a';
[x,fs] = audioread(whaleFile);
x=x(:,1);
subplot(211);
plot(x)
xlabel('Sample Number')
ylabel('Amplitude')
% % 音频剪切
% moan = x(3.525e5:4.482e5);
% t = 10*(0:1/fs:(length(moan)-1)/fs);
% plot(t,moan)
% xlabel('Time (seconds)')
% ylabel('Amplitude')
% xlim([0 t(end)])
% 功率频谱，在一些使用 fft 处理大量数据的应用中，通常需要调整输入，使样本数量为 2 的幂。这样可以大幅提高变换计算的速度
% m = length(moan);       % original sample length
% n = pow2(nextpow2(m));  % transform length
% y = fft(moan,n);        % DFT of signal
% 
% f = (0:n-1)*(fs/n)/10;
% power = abs(y).^2/n;      
% subplot(211);
% plot(f(1:floor(n/2)),power(1:floor(n/2)))
% xlabel('Frequency')
% ylabel('Power')
%chirp频谱
moan=x(:,1);%我这里假设你的声音是双声道，我只取单声道作分析，如果你想分析另外一个声道，请改成y=y(:,2)
[S, F, T] = spectrogram(moan, hanning(1024), 512, 1024, fs);
tt=T;
Ff=F/1000;%将频率轴变为以kHz为单位
subplot(212);
imagesc(tt, Ff, log10(abs(S)));
set(gca, 'YDir', 'normal');
xlabel('Time(s)');
ylabel('Frequency(kHz)');
title('air时频谱')
% % % specgram(moan,512,fs,100)%语谱图函数
% % % xlabel('时间(s)')
% % % ylabel('频率(Hz)')
% % % title('“概率”语谱图')
