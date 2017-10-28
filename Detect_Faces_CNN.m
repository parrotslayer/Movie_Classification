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
% This is i for when you loop
name_ID = 1;
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

%% Crop faces and resize
[rows,cols] = size(bboxes);
for i = 1:rows
    Movie_Faces{i} = I_BW(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3));
    % Has to be resized into the size of the faces used for training
    Movie_Faces{i} = imresize(Movie_Faces{i},[64 64]);
    % Save movie face
    str = strcat(int2str(name_ID),'_face_',int2str(i),'.jpg');
    file_names{i} = str;
    % image to write
    imwrite(Movie_Faces{i},str);
end

%% Convert to ImageDataStore
labels = ones(rows,1)*name_ID;
FaceData = imageDatastore(file_names);

%% Classify the Images in the Test Data and Compute Accuracy
% Load convnet
load('convnet_100epoch.mat')

% Run the trained network on the test set that was not used to train the
% network and predict the image labels (digits).
label_pred_cat = classify(convnet,FaceData);

% Convert catagorical to cell
for i = 1:rows
    if label_pred_cat(i) == categorical(0)
        label_pred(i) = 0;
    else
        label_pred(i) = 1;
    end
end
    
% Annotate detected faces.
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_pred);
figure, imshow(IFaces), title('Detected Faces Using CNN 0 = F, 1 = M');