close all
clc
clear all

load('Eigen_Weights_All.mat');
load('wiki.mat');
load('path_0_2.mat');

gender = zeros(length(path_images),1);
for i = 1:length(path_images)
    index = path_images(i);
    image_path_names{i} = strcat(int2str(i),'_BW.jpg');
    gender(i) = wiki.gender(index);
end

%% Divide dataset into training and validation
data_eigen = [InImWeight_All,e_All];
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
save('Mdl_0_2_weights.mat','Mdl');

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

% Assess whether true positive or not
true_pos = 0;
conf = zeros(2,2);
for i = 1:length(labels_test)
    if label_pred(i) == labels_test(i)
        if label_pred(i) == 0
            conf(1,1) = conf(1,1) + 1;
        else
            conf(2,2) = conf(2,2) + 1;
        end
    else
        % pred 0 true 1
        if label_pred(i) == 0
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
