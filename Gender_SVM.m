close all
clc
%clear all

load('eigen_faces_00.mat');
load('omega_00.mat');
load('wiki.mat');

%% Get labels for genders
%  0 for female and 1 for male, NaN if unknown

gender = NaN(length(path_00),1);
% Get metadata from the training set
for i = 1:length(path_00)
    index = path_00(i);
    % Get the gender
    gender(i) = wiki.gender(index);
end

[rows,cols,num_images] = size(data_train);
% randomise the data
new_index = randperm(num_images);
data_train = data_train(:,:,new_index);
labels_train = labels_train(new_index);

%take only a bit of the data
data_sample = 5000;
data_train = data_train(:,:,1:data_sample);
labels_train = labels_train(1:data_sample);

[rows,cols,num_images] = size(data_train);

num_folds = 10;
train_folds = 5;
samples = num_images/num_folds;

%make training data
im_sub = data_train(:,:,1:samples*train_folds);
labels_sub = labels_train(1:samples*train_folds,:);

% make testing data
im_sub2 = data_train(:,:,1:samples*num_folds);
labels_sub2 = labels_train(1:samples*num_folds,:);
im_sub2 = data_test;
labels_sub2 = labels_test;

%% Make the machine learning model or just load it
% Make the model

make_model = 1;

if make_model == 1
    for i = 1:length(trainingSet.Files)
        %for i = 1:5
        %extract features using extractHOG
        clear image
        image = trainingSet.readimage(i);
        [featureVector_HOG,hogVisualization] = extractHOGFeatures(image);
        %normalise the feature vector
        featureVector_HOG = featureVector_HOG./max(featureVector_HOG);
        
        [rows,cols,pages] = size(image);
        % Get raw pixel data of colour
        k = 1;
        KS = 0;
        for r = 1:rows
            for c = 1:cols
                for p = 1:pages
                    %subsample the image data by a factor KS
                    if KS == 4
                        %make 1D array of the colour image
                        featureVector_col(k) = image(r,c,p);
                        k = k + 1;
                        KS = 0;
                    else
                        KS = KS + 1;
                    end                    
                end
            end
        end
        %normalise the featureVector and convert to double
        featureVector_col = double(featureVector_col./max(featureVector_col));
        
        % Use historgram data over the RGB channels. Could do over HSV if
        % wanted to.
        % Repeat for R, G, B channels with X bins each
        num_bins = 10000;
        [counts_R,bins] = imhist(image(:,:,1),num_bins);
        [counts_G,bins] = imhist(image(:,:,2),num_bins);
        [counts_B,bins] = imhist(image(:,:,3),num_bins);
        featureVector_Hist = [counts_R',counts_G',counts_B'];
        
        %normalise the featureVector along ALL the 3 channels.
        % Could normalise for each channel seperately?
        featureVector_Hist = double(featureVector_Hist./max(featureVector_Hist));
        
        % Concatenate the different feature vectors
        featureVector(i,:) = [featureVector_HOG, featureVector_col,featureVector_Hist];
        
    end
    % Train an ECOC multiclass model using the default options. SVM
    Mdl = fitcecoc(featureVector,trainingSet.Labels);
    % save the generated model
    save('Mdl.mat','Mdl');
else
    load('Mdl.mat')
end

%% Pass in the testing data
% take first remaining folds and classify the data
for i = 1:length(testSet.Files)
    %extract features using extractHOG
    clear image
    clear featureVector_col
    clear featureVector_Hist
    image = testSet.readimage(i);
    [featureVector_HOG,hogVisualization] = extractHOGFeatures(image);
    %normalise the feature vector
    featureVector_HOG = featureVector_HOG./max(featureVector_HOG);
    
    [rows,cols,pages] = size(image);
    % Get raw pixel data of colour
    k = 1;
    KS = 0;
    for r = 1:rows
        for c = 1:cols
            for p = 1:pages
                %subsample the image data by a factor
                if KS == 4
                    %make 1D array of the colour image
                    featureVector_col(k) = image(r,c,p);
                    k = k + 1;
                    KS = 0;
                else
                    KS = KS + 1;
                end
            end
        end
    end
    
    %normalise the featureVector and convert to double
    featureVector_col = double(featureVector_col./max(featureVector_col));
    
    % Use historgram data over the RGB channels. Could do over HSV if
    % wanted to.
    % Repeat for R, G, B channels with X bins each
    num_bins = 10000;
    [counts_R,bins] = imhist(image(:,:,1),num_bins);
    [counts_G,bins] = imhist(image(:,:,2),num_bins);
    [counts_B,bins] = imhist(image(:,:,3),num_bins);
    featureVector_Hist = [counts_R',counts_G',counts_B'];
    
    %normalise the featureVector along ALL the 3 channels.
    % Could normalise for each channel seperately?
    featureVector_Hist = double(featureVector_Hist./max(featureVector_Hist));
    
    % Concatenate the different feature vectors
    featureVector2(i,:) = [featureVector_HOG, featureVector_col,featureVector_Hist];
           
end

% Pass features into predict. Returns vector with predicted
label_pred = predict(Mdl,featureVector2);

%% Assess whether true positive or not
true_pos = 0;
conf = zeros(7,7);
for i = 1:length(testSet.Files)
    if label_pred(i) == testSet.Labels(i)
        true_pos = true_pos + 1;
        %increment confusion matrix
        i_pred = label2number(char(label_pred(i)));
        i_true = label2number(char(testSet.Labels(i)));
        conf(i_pred,i_true) = conf(i_pred,i_true) + 1;
    else
        %figure
        %imshow(testSet.readimage(i))
        %title(['P = ',char(label_pred(i)),' T = ',char(testSet.Labels(i))])
        %increment confusion matrix
        i_pred = label2number(char(label_pred(i)));
        i_true = label2number(char(testSet.Labels(i)));
        conf(i_pred,i_true) = conf(i_pred,i_true) + 1;
    end
end