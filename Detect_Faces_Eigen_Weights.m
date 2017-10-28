%% Detect Faces in an Image Using the Frontal Face Classification Model
clc
close all
clear all

% Load eigen face stuff
load('Eigen_Required_Variables.mat')
load('Eigen_Weights_All.mat')

% Get the Movie Poster
addpath(genpath('Action'))
name = 'allegiant_2016';
str=strcat(name,'.jpg');

% Create a detector object.
faceDetector = vision.CascadeObjectDetector;

% Read input image.
I = imread(str);

% Convert to Greyscale
I_BW = rgb2gray(I);

% Detect faces.
bboxes = step(faceDetector, I);

% Annotate detected faces.
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');
figure, imshow(IFaces), title('Detected faces');

%% Preconditioning for SVM
% Crop faces and resize
[rows,cols] = size(bboxes);
for i = 1:rows
    Movie_Faces{i} = I_BW(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3));
    % Has to be resized into the size of the faces used for training
    Movie_Faces{i} = imresize(Movie_Faces{i},[256 256]);
end

%% Compute the eigen faces
[Weight_All,e_All] = Get_Eigen_Weights_RawData(Movie_Faces);

%% Apply ML Model Using Omega
% load the generated model
load('Mdl_0_2_weights.mat')
featureVector2 = [Weight_All,e_All];

% Pass features into predict. Returns vector with predicted
label_pred = predict(Mdl,featureVector2);

% Annotate detected faces.
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_pred);
figure, imshow(IFaces), title('Using Eigen Faces F = 0, M = 1');