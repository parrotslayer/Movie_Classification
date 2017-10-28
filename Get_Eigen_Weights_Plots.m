%% New Image
% Acquire new image
% Note: It should have the same size as the ones in your training set. 
clc
clear all

% string_name = 'Eigen_Workspace_All.mat';
% load(string_name, 'um');
% load(string_name, 'ustd');
% load(string_name, 'u');
% load(string_name, 'irow');
% load(string_name, 'icol');
% load(string_name, 'temp');
% load(string_name, 'm');
% load(string_name, 'M');
% load(string_name, 'omega');
% save('Eigen_Required_Variables.mat')

load('Eigen_Required_Variables.mat')

%%
addpath(genpath('Faces_0_2_Cropped_BW'));
file_name = '1_BW.jpg';
InputImage = imread(file_name);
figure(5)
subplot(1,2,1)
imshow(InputImage); colormap('gray');title('Input image','fontsize',18)
InImage=reshape(double(InputImage)',irow*icol,1);  
temp=InImage;
me=mean(temp);
st=std(temp);
temp=(temp-me)*ustd/st+um;
NormImage = temp;
Difference = temp-m;

p = [];
aa=size(u,2);
for i = 1:aa
    pare = dot(NormImage,u(:,i));
    p = [p; pare];
end
ReshapedImage = m + u(:,1:aa)*p;    %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';
%show the reconstructed image.
subplot(1,2,2)
imagesc(ReshapedImage); colormap('gray');
title('Reconstructed image','fontsize',18)

InImWeight = [];
for i=1:size(u,2)
    t = u(:,i)';
    WeightOfInputImage = dot(t,Difference');
    InImWeight = [InImWeight; WeightOfInputImage];
end

ll = 1:M;
figure
stem(ll,InImWeight,'Marker','none')
title('Weight of Input Face')

% Find Euclidean distance
e=[];
for i=1:size(omega,2)
    q = omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e = [e mag];
end

kk = 1:size(e,2);
figure
stem(kk,e,'Marker','none')
title('Eucledian distance of input image')

MaximumValue=max(e)
MinimumValue=min(e)