% %%������FRM1��test.m���У�����Ϊ���ص���
clear;
clc;
[filename,pathname]=uigetfile('����ָRIO.bmp','E:\������ͨ��ѧ\һЩ����\blood flow of finger vein\data\');%��ѡ��Ҫ�����ͼƬ
path=[pathname,filename];
% path1 = 'E:\������ͨ��ѧ\һЩ����\blood flow of finger vein\data\�Ҵ�Ĵָ.bmp';
I=imread(path);
Origimg = rgb2gray(I); %ת��Ϊ�Ҷ�ͼ
% imshow(Origimg);
img=imresize(Origimg,[40 40]);%�ı�ͼ��ߴ�Ĵ�СΪ532*784
imshow(img);
%�趨�洢���ļ���
path2='E:\������ͨ��ѧ\һЩ����\blood flow of finger vein\data\savaPicture';
pathfile=fullfile(path2,filename);
imwrite(img,pathfile,'jpg');