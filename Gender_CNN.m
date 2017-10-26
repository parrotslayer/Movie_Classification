%clc
clear all
close all

load('path_0_2.mat')
load('wiki.mat')
addpath(genpath('Faces_0_2_Cropped_BW'));

gender = zeros(length(path_images),1);
for i = 1:length(path_images)
    index = path_images(i);
    image_path_names{i} = strcat(int2str(i),'_BW.jpg');
    gender(i) = wiki.gender(index);
end

%% Load Data
FaceData = imageDatastore(image_path_names,'Labels',categorical(gender));

% Display some of the images in the datastore. 
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(FaceData.Files{i});
end

% Check the number of images in each category. 
CountLabel = FaceData.countEachLabel;

%% Specify Training and Test Sets
% Divide the data into training and test sets, so that each category in the
% training set has 750 images and the test set has the remaining images
% from each label.
ratio = 0.5;
trainingNumFiles = round(length(FaceData)*ratio);
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(FaceData, ...
				trainingNumFiles,'randomize'); 

%% Define the Network Layers
% Define the convolutional neural network architecture. 
layers = [imageInputLayer([256 256 1])
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(2)
          softmaxLayer
          classificationLayer()];  

% Specify the Training Options
options = trainingOptions('sgdm','MaxEpochs',500, ...
	'InitialLearnRate',0.0001);  

%% Train the Network Using Training Data
% Train the network you defined in layers, using the training data and the
% training options you defined in the previous steps.
convnet = trainNetwork(trainDigitData,layers,options);

%% Classify the Images in the Test Data and Compute Accuracy
% Run the trained network on the test set that was not used to train the
% network and predict the image labels (digits).
YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;

%% 
% Calculate the accuracy. 
accuracy = sum(YTest == TTest)/numel(TTest)   

%%
% Accuracy is the ratio of the number of true labels in the test data
% matching the classifications from classify, to the number of images in
% the test data. In this case about 98% of the digit estimations match the
% true digit values in the test set.