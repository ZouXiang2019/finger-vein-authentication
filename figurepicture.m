%������ֵ
whaleFile = 'air1+finger1.m4a';
[x,fs] = audioread(whaleFile);
x=x(:,1);
subplot(211);
plot(x)
xlabel('Sample Number')
ylabel('Amplitude')
% % ��Ƶ����
% moan = x(3.525e5:4.482e5);
% t = 10*(0:1/fs:(length(moan)-1)/fs);
% plot(t,moan)
% xlabel('Time (seconds)')
% ylabel('Amplitude')
% xlim([0 t(end)])
% ����Ƶ�ף���һЩʹ�� fft ����������ݵ�Ӧ���У�ͨ����Ҫ�������룬ʹ��������Ϊ 2 ���ݡ��������Դ����߱任������ٶ�
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
%chirpƵ��
moan=x(:,1);%������������������˫��������ֻȡ����������������������������һ����������ĳ�y=y(:,2)
[S, F, T] = spectrogram(moan, hanning(1024), 512, 1024, fs);
tt=T;
Ff=F/1000;%��Ƶ�����Ϊ��kHzΪ��λ
subplot(212);
imagesc(tt, Ff, log10(abs(S)));
set(gca, 'YDir', 'normal');
xlabel('Time(s)');
ylabel('Frequency(kHz)');
title('airʱƵ��')
% % % specgram(moan,512,fs,100)%����ͼ����
% % % xlabel('ʱ��(s)')
% % % ylabel('Ƶ��(Hz)')
% % % title('�����ʡ�����ͼ')
