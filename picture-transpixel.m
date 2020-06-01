% %%真正在FRM1的test.m运行，这里为像素调整
clear;
clc;
[filename,pathname]=uigetfile('左中指RIO.bmp','E:\西安交通大学\一些尝试\blood flow of finger vein\data\');%请选择要处理的图片
path=[pathname,filename];
% path1 = 'E:\西安交通大学\一些尝试\blood flow of finger vein\data\右大拇指.bmp';
I=imread(path);
Origimg = rgb2gray(I); %转换为灰度图
% imshow(Origimg);
img=imresize(Origimg,[40 40]);%改变图像尺寸的大小为532*784
imshow(img);
%设定存储的文件夹
path2='E:\西安交通大学\一些尝试\blood flow of finger vein\data\savaPicture';
pathfile=fullfile(path2,filename);
imwrite(img,pathfile,'jpg');