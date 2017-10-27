
clear all
close all
clc

rootFolder = fullfile(pwd,'dataset');
current_img = imread(fullfile(rootFolder, 'desert', '00000005.jpg'));

% HOG features extraction
disp('Processing HOG features ...')

figure;
imshow(current_img);
title('Original Image');
% Color Histogram features extration
figure;
temp = rgb2hsv(current_img);
imshow(temp);
title('Image in HSV');
figure
imhist(temp(:,:,1),30);
title('Intensity Histogram of in HSV image');
figure
imhist(current_img(:,:,1),30);
title('Intensity Histogram of in Original image');

current_img = imbinarize(rgb2gray(current_img));
[featureVector, hogVisualization] = extractHOGFeatures(current_img)
figure;
imshow(current_img); hold on;
plot(hogVisualization);   
title('HOGS extraction of Original image');

rootFolder = fullfile(pwd,'dataset');

image_path=fullfile(rootFolder, 'ball_pit', '00000158.jpg')

img = imread(fullfile(image_path));
    load categoryClassifierFull.mat;
    %load categoryClassifierCrop.mat;
    disp('Training');

    categories = {'ball_pit','desert','park','road','sky','snow','urban'};

    [labelIdx, scores] = predict(categoryClassifier, img);
    class_labels = categoryClassifier.Labels(labelIdx)
    
    imgSet=imageSet(rootFolder,'recursive');
    [trainingSet, validationSet] = partition(imgSet, 0.7, 'randomize')

    confMatrix = evaluate(categoryClassifier, validationSet);
    
[class_labels] = assign2_sceneclassifier(fullfile(rootFolder, 'ball_pit', '00000158.jpg'))
