%% Detect Faces in an Image Using the Frontal Face Classification Model
clc
close all
clear all

% Copyright 2015 The MathWorks, Inc.
addpath(genpath('Action'))
name = 'Baby_driver_2017';
str=strcat(name,'.jpg');

% Create a detector object.
faceDetector = vision.CascadeObjectDetector;

% Read input image.
I = imread(str);

% Detect faces.
bboxes = step(faceDetector, I);

% Annotate detected faces.
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');
figure, imshow(IFaces), title('Detected faces');

%% Preconditioning for SVM
% Crop faces and resize
[rows,cols] = size(bboxes);
for i = 1:rows
    Movie_Faces{i} = I(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3));
    % Has to be resized into the size of the faces used for training
    Movie_Faces{i} = imresize(Movie_Faces{i},[256 256]);
    
    %figure
    %imshow(Movie_Faces{i})
end

% Compute the eigen faces
eigen_faces_movie = Get_Eigen_Face(Movie_Faces);
figure
imshow(eigen_faces_movie(:,:,1))
title('Eigen Face')

%% Apply ML Model

% load the generated model
load('Mdl_0_2.mat');

%% Pass in the data
[rows,cols,pages] = size(eigen_faces_movie);
for i = 1:pages
    % Get raw pixel data of colour
    k = 1;
    for r = 1:rows
        for c = 1:cols
            %make 1D array of the eigen faces
            featureVector2(i,k) = eigen_faces_movie(r,c,i);
            k = k + 1;
        end
    end
end

% Pass features into predict. Returns vector with predicted
label_pred = predict(Mdl,featureVector2);

% Annotate detected faces.
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_pred);
figure, imshow(IFaces), title('Detected faces');