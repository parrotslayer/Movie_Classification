%% Detect Faces in an Image Using the Frontal Face Classification Model
clc
% Copyright 2015 The MathWorks, Inc.
addpath(genpath('Action'))
name = 'king_arthur_2017';
name = 'allegiant_2016';
str=strcat(name,'.jpg');

%% Create a detector object.
    faceDetector = vision.CascadeObjectDetector; 
    
%% Read input image.
    I = imread(str);
    
%% Detect faces. 
    bboxes = step(faceDetector, I);
    
%% Annotate detected faces.
   IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');   
   figure, imshow(IFaces), title('Detected faces'); 
   
   
