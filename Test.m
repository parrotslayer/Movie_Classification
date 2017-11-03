
clear all
close all
clc

rootFolder = fullfile(pwd,'Movie_Classification');
current_img = imread(fullfile(rootFolder, 'Horror', 'Green_Room_2015.jpg'));

% HOG features extraction
disp('Processing HOG features ...')


figure;
imshow(current_img);
title('Original Image');

figure;
imshow(current_img);
[featureVector, hogVisualization] = extractHOGFeatures(current_img);
hold on;
plot(hogVisualization);
title('HOGS extraction of RBG image')

% RGB
figure
imhist(current_img(:,:,1),500);
title('Intensity Histogram of Red image');
figure
imhist(current_img(:,:,2),500);
title('Intensity Histogram of Green image');
figure
imhist(current_img(:,:,3),500);
title('Intensity Histogram of Blue image');




% Color Histogram features extration
figure;
temp = rgb2hsv(current_img);
imshow(temp);
title('Image in HSV');
figure
imhist(temp(:,:,1),500);
title('Intensity Histogram of in Hue image');
figure
imhist(temp(:,:,2),500);
title('Intensity Histogram of in Saturation image');
figure
imhist(temp(:,:,3),500);
title('Intensity Histogram of in Lightness image');

% Detect SURF features extration
temp_img = rgb2gray(current_img);
disp('Processing SURF features ...')
points = detectSURFFeatures(temp_img);
figure;
imshow(current_img); hold on;
plot(points.selectStrongest(100));
title('SURF features of image');

% Detect MSER features extration
disp('Processing MSER features ...')


regionsObj = detectMSERFeatures(temp_img);
% Detect SURF point
[MSER_features, validPtsObj] = extractFeatures(temp_img, regionsObj);
figure
imshow(current_img); hold on;
plot(validPtsObj.selectStrongest(100),'showOrientation',true);
title('MSER features of image');




current_img = imbinarize(rgb2gray(current_img));
[featureVector, hogVisualization] = extractHOGFeatures(current_img);
figure;
imshow(current_img); hold on;
plot(hogVisualization);   
title('HOGS extraction of BW image');






% rootFolder = fullfile(pwd,'dataset');
% 
% image_path=fullfile(rootFolder, 'ball_pit', '00000158.jpg')
% 
% img = imread(fullfile(image_path));
%     load categoryClassifierFull.mat;
%     %load categoryClassifierCrop.mat;
%     disp('Training');
% 
%     categories = {'ball_pit','desert','park','road','sky','snow','urban'};
% 
%     [labelIdx, scores] = predict(categoryClassifier, img);
%     class_labels = categoryClassifier.Labels(labelIdx)
%     
%     imgSet=imageSet(rootFolder,'recursive');
%     [trainingSet, validationSet] = partition(imgSet, 0.7, 'randomize')
% 
%     confMatrix = evaluate(categoryClassifier, validationSet);
%     
% [class_labels] = assign2_sceneclassifier(fullfile(rootFolder, 'ball_pit', '00000158.jpg'))
