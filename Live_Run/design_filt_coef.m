clc
clear all
close all

%%
%Design filter for live run filter
% x_n=load('new_breath.csv');
% x_n=load('breath_depth.csv');
x_n=load('test1.csv');
% x_n=load('test2.csv');
figure(1)
plot(x_n);

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

cut_off=[0.04 0.4];
order=12;
h=fir1(order,cut_off);

con=filter(h,1,x);
figure(3)
plot(con)


