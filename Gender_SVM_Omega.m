close all
clc
%clear all

load('omega_0_2.mat');
load('eigen_faces_0_2.mat');
load('wiki.mat');
load('path_0_2.mat');
%% Get labels for genders
%  0 for female and 1 for male, NaN if unknown
gender = NaN(length(path_images),1);
% Get metadata from the training set
for i = 1:length(path_images)
    index = path_images(i);
    % Get the gender
    gender(i) = wiki.gender(index);
end

%% Divide dataset into training and validation
data_eigen = omega;
labels_eigen = gender;

[num_images,cols] = size(data_eigen);

% randomise the indexes
new_index = randperm(num_images);

% reorder images by randomised indexes
data_eigen = data_eigen(new_index,:);
labels_eigen = labels_eigen(new_index);

fold_ratio = 0.5;
split = round(fold_ratio*num_images);

%make training data
data_train = data_eigen(1:split,:);
labels_train = labels_eigen(1:split);

% make testing data
data_test = data_eigen(split+1:end,:);
labels_test = labels_eigen(split+1:end);

%% Make the model
featureVector = data_test;

% Train an ECOC multiclass model using the default options.
Mdl = fitcecoc(featureVector,labels_train);

% save the generated model
save('Mdl_0_2_omega.mat','Mdl');

%% Pass in the testing data
featureVector2 = data_test;

% Pass features into predict. Returns vector with predicted
label_pred = predict(Mdl,featureVector2);

%% Verification
count_true = 0;
for i = 1:length(labels_test)
   if label_pred(i) == labels_test(i)
       count_true = count_true + 1;
   end    
end

disp(['Training Set = ', num2str(fold_ratio*100),' %'])
count_true_pc = count_true/length(labels_test)*100;
disp(['Accuracy = ', num2str(count_true_pc),' %'])

