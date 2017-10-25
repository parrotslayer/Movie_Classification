close all
clc
clear all

load('eigen_faces_00.mat');
load('wiki.mat');
load('path_00.mat');
%% Get labels for genders
%  0 for female and 1 for male, NaN if unknown

gender = NaN(length(path_00),1);
% Get metadata from the training set
for i = 1:length(path_00)
    index = path_00(i);
    % Get the gender
    gender(i) = wiki.gender(index);
end

%% Divide dataset into training and validation
data_eigen = eigen_faces;
labels_eigen = gender;

[rows,cols,num_images] = size(data_eigen);

% randomise the indexes
new_index = randperm(num_images);

% reorder images by randomised indexes
data_eigen = data_eigen(:,:,new_index);
labels_eigen = labels_eigen(new_index);

fold_ratio = 0.5;
split = round(fold_ratio*num_images);

%make training data
im_train = data_eigen(:,:,1:split);
labels_train = labels_eigen(1:split);

% make testing data
im_test = data_eigen(:,:,split+1:end);
labels_test = labels_eigen(split+1:end);

%% Make the model
% gets first N folds and classify the data
[rows,cols,pages] = size(eigen_faces);
for i = 1:length(labels_train)
    % Get raw pixel data of colour
    k = 1;
    for r = 1:rows
        for c = 1:cols
            %make 1D array of the greyscale image
            featureVector(i,k) = im_train(r,c,i);
            k = k + 1;
        end
    end
end

% Train an ECOC multiclass model using the default options.
Mdl = fitcecoc(featureVector,labels_train);

% save the generated model
save('Mdl.mat','Mdl');

%% Pass in the testing data
% take first remaining folds and classify the data
for i = 1:length(labels_test)
    % Get raw pixel data of colour
    k = 1;
    for r = 1:rows
        for c = 1:cols
            %make 1D array of the greyscale image
            featureVector2(i,k) = im_test(r,c,i);
            k = k + 1;
        end
    end
end

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

