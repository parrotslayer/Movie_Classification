clear all
close all
clc

load('path_0_2.mat')
load('wiki.mat')
addpath(genpath('Faces_0_2_Cropped_BW_64'));

gender = zeros(length(path_images),1);
for i = 1:length(path_images)
    index = path_images(i);
    image_path_names{i} = strcat(int2str(i),'_BW.jpg');
    gender(i) = wiki.gender(index);
end

%% Load Data
FaceData = imageDatastore(image_path_names,'Labels',categorical(gender));

% Check the number of images in each category. 
CountLabel = FaceData.countEachLabel;

%% Specify Training and Test Sets
% Divide the data into training and test sets, so that each category in the
% training set has 750 images and the test set has the remaining images
% from each label.
ratio = 0.5;
[trainDigitData,testDigitData] = splitEachLabel(FaceData,ratio,'randomized'); 

%% Define the Network Layers
% Define the convolutional neural network architecture. 
layers = [imageInputLayer([64 64 1])
          convolution2dLayer(3,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(2)
          softmaxLayer
          classificationLayer()];  

% Specify the Training Options
options = trainingOptions('sgdm','MaxEpochs',100, ...
	'InitialLearnRate',0.0001);  

%% Train the Network Using Training Data
% Train the network you defined in layers, using the training data and the
% training options you defined in the previous steps.
convnet = trainNetwork(trainDigitData,layers,options);
%% save
save('convnet_100epoch.mat','convnet');

%% Classify the Images in the Test Data and Compute Accuracy
% Run the trained network on the test set that was not used to train the
% network and predict the image labels (digits).
YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;

%% Verification

accuracy = sum(YTest == TTest)/numel(TTest);
zero0 = categorical(0);
one1 = categorical(1);

label_pred = YTest;

labels_test = testDigitData.Labels;
fold_ratio = ratio;

count_true = 0;
for i = 1:length(labels_test)
   if label_pred(i) == labels_test(i)
       count_true = count_true + 1;
   end    
end

% Assess whether true positive or not
true_pos = 0;
conf = zeros(2,2);
for i = 1:length(labels_test)
    if label_pred(i) == labels_test(i)
        if label_pred(i) == zero0
            conf(1,1) = conf(1,1) + 1;
        else
            conf(2,2) = conf(2,2) + 1;
        end
    else
        % pred 0 true 1
        if label_pred(i) == zero0
            conf(1,2) = conf(1,2) + 1;
        else
            conf(2,1) = conf(2,1) + 1;
        end
    end
end

% Show evaluation
disp(['Fold Ratio = ',num2str(fold_ratio*100),' %']);
true_pos_rate = count_true/length(labels_test)*100;
disp(['True Positive Rate = ', num2str(true_pos_rate),' %'])

precision = zeros(1,2);
recall = zeros(1,2);
for i = 1:2
    %calc precision (across)
    precision(i) = conf(i,i)/sum(conf(i,:));
    %calc recall (down)
    recall(i) = conf(i,i)/sum(conf(:,i));
end

F1 = 2*(precision.*recall)/(precision+recall)*100;
disp(['F1 Score = ', num2str(F1),' %'])
disp('Confusion Matrix with Recall and Precision')

results = zeros(3);
results(2:3,1) = precision;
results(1,2:3) = recall;
results(2:3,2:3) = conf;
disp(results)