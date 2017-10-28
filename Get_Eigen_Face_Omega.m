%% Takes in the cropped faces resized
% Needs to be [256x256]
function omega = Get_Eigen_Face_Omega(cropped_face_resized)
%import u
load('eigen_u.mat');

% number of images on your training set.
M=length(cropped_face_resized);

%Chosen std and mean.
%It can be any number that it is close to the std and mean of most of the images.
um=100;
ustd=80;

%read and show images(bmp);
S=[];   %img matrix
for i=1:M
    img = uint8(cropped_face_resized{i});
    % Convert RGB to Greyscale if needed
    [rows,cols,pages] = size(cropped_face_resized{i});
    if pages > 1
        img = rgb2gray(img);
    end
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
end

%% Compute Eigen Faces
%mean image;
m=mean(S,2);   %obtains the mean of each row instead of each column
tmimg=uint8(m);   %converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,icol,irow);    %takes the N1*N2x1 vector and creates a N2xN1 matrix
img=img';       %creates a N1xN2 matrix by transposing the image.
%figure;
%imshow(img);
%title('Mean Image','fontsize',18)

% Change image for manipulation
dbx=[];   % A matrix
for i=1:M
    temp=double(S(:,i));
    dbx=[dbx temp];
end


%% Find the weight of each face in the training set.
omega = [];
for h=1:size(dbx,2)
    WW=[];    
    for i=1:size(u,2)
        t = u(:,i)';    
        WeightOfImage = dot(t,dbx(:,h)');
        WW = [WW; WeightOfImage];
    end
    omega = [omega WW];
end

end