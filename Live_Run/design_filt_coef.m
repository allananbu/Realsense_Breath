clc
clear all
close all

%%
%Design filter for live run filter
% x_n=load('new_breath.csv');
% x_n=load('breath_depth.csv');
% x_n=load('Tuesday.csv');
% x_n=load('test2.csv');
% x_n=load('slow_breath.csv');
% x_n=load('slow_breath_1.csv');
x_n=load('update_filt.csv');

figure(1)
plot(x_n)

Fs=20;
x=x_n-mean(x_n);%Eliminate DC
nfft=length(x);
nfft2=2.^nextpow2(nfft);
fy=fft(x,nfft2);
fy=abs(fy(1:nfft2/2));
xfft=Fs.*(0:nfft2/2-1)/nfft2;
[v,h]=max(fy);
figure(2)
plot(xfft,(fy/max(fy)))

%Define cut off frequency and calculate filter coefficients
cut_off=[0.02 0.3];
order=16;
h=fir1(order,cut_off);

con=filtfilt(h,1,x);
figure(3)
plot(con)


