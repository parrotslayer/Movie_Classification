% Face recognition 
%clear all
close all
clc

%% Load in the Images
% load in cropped images
load('cropped_face_00.mat');
load('wiki.mat');
load('path_00.mat');

% number of images on your training set.
M=length(cropped_face_resized);

%M = 40;

%Chosen std and mean. 
%It can be any number that it is close to the std and mean of most of the images.
um=100;
ustd=80;

%read and show images(bmp);
S=[];   %img matrix
%figure(1);
for i=1:M
%     str=strcat(int2str(i),'.bmp');   %concatenates two strings that form the name of the image
%     eval('img=imread(str);');
    img = uint8(cropped_face_resized{i});
    % Convert RGB to Greyscale if needed
    [rows,cols,pages] = size(cropped_face_resized{i});
    if pages > 1
        img = rgb2gray(img);
    end
    %subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    %imshow(img)
    %if i==3
    %    title('Training set','fontsize',18)
    %end
    %drawnow;
    [irow icol]=size(img);    % get the number of rows (N1) and columns (N2)
    temp=reshape(img',irow*icol,1);     %creates a (N1*N2)x1 matrix
    S=[S temp];         %X is a N1*N2xM matrix after finishing the sequence
                        %this is our S
end


%Here we change the mean and std of all images. We normalize all images.
%This is done to reduce the error due to lighting conditions.
for i=1:size(S,2)
    temp=double(S(:,i));
    m=mean(temp);
    st=std(temp);
    S(:,i)=(temp-m)*ustd/st+um;
end

%% show normalized images
%figure(2);
for i=1:M
    str=strcat(int2str(i),'.jpg');
    img=reshape(S(:,i),icol,irow);
    img=img';
    %****************** This prints out images ************************
    %eval('imwrite(img,str)');
    
    %subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    %imshow(img)
    %drawnow;
    %if i==3
    %    title('Normalized Training Set','fontsize',18)
    %end
end

%% Compute Eigen Faces
%mean image;
m=mean(S,2);   %obtains the mean of each row instead of each column
tmimg=uint8(m);   %converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,icol,irow);    %takes the N1*N2x1 vector and creates a N2xN1 matrix
img=img';       %creates a N1xN2 matrix by transposing the image.
figure;
imshow(img);
title('Mean Image','fontsize',18)

% Change image for manipulation
dbx=[];   % A matrix
for i=1:M
    temp=double(S(:,i));
    dbx=[dbx temp];
end

%Covariance matrix C=A'A, L=AA'
A=dbx';
L=A*A';
% vv are the eigenvector for L
% dd are the eigenvalue for both L=dbx'*dbx and C=dbx*dbx';
[vv dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
    if(dd(i,i)>1e-4)
        v=[v vv(:,i)];
        d=[d dd(i,i)];
    end
 end
 
 %sort,  will return an ascending sequence
 [B index]=sort(d);
 ind=zeros(size(index));
 dtemp=zeros(size(index));
 vtemp=zeros(size(v));
 len=length(index);
 for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    vtemp(:,ind(i))=v(:,i);
 end
 d=dtemp;
 v=vtemp;


%Normalization of eigenvectors
 for i=1:size(v,2)       %access each column
   kk=v(:,i);
   temp=sqrt(sum(kk.^2));
   v(:,i)=v(:,i)./temp;
end

%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
    temp=sqrt(d(i));
    u=[u (dbx*v(:,i))./temp];
end

%Normalization of eigenvectors
for i=1:size(u,2)
   kk=u(:,i);
   temp=sqrt(sum(kk.^2));
	u(:,i)=u(:,i)./temp;
end


%% show eigenfaces;
eigen_faces = zeros(256,256,size(u,2));
%figure(4);
for i=1:size(u,2)
    img=reshape(u(:,i),icol,irow);
    img=img';
    img=histeq(img,255);
    %subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    %imshow(img)
    %drawnow;
    eigen_faces(:,:,i) = img;
    %if i==3
    %    title('Eigenfaces','fontsize',18)
    %end
end

%%
save('eigen_faces_00.mat','eigen_faces');

%% Find the weight of each face in the training set.
% omega = [];
% for h=1:size(dbx,2)
%     WW=[];    
%     for i=1:size(u,2)
%         t = u(:,i)';    
%         WeightOfImage = dot(t,dbx(:,h)');
%         WW = [WW; WeightOfImage];
%     end
%     omega = [omega WW];
% end