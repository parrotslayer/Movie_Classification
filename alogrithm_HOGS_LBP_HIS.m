%% Load up all Images


clear all
close all
clc

%% File Opening

root_folder = 'Movie_Classification';
horror_dir = {[root_folder filesep 'horror' filesep '*.jpg']};
action_dir = {[root_folder filesep 'action' filesep '*.jpg']};
romance_dir = {[root_folder filesep 'romance' filesep '*.jpg']};
comedy_dir = {[root_folder filesep 'comedy' filesep '*.jpg']};

classes = [horror_dir,action_dir,romance_dir,comedy_dir];
folder_dir = {'horror','action','romance','comedy'};

%% Data Pre-processing

num_file = 1;

% Loop through each category
for i = 1:length(classes)
    dirs = dir(classes{i});      
    nfiles = length(dirs);
    % Loop through each image
    for n=1:nfiles
       currentfile = [root_folder, filesep, folder_dir{i}, filesep, dirs(n).name];
       currentimage = imread(currentfile);
       img(:,:,:,num_file) = imresize((currentimage),[1000,674]);
       category(num_file) = i;
       num_file = num_file + 1;
    end
end

num_file = num_file - 1;
category = category';

%% Data Augmentation
disp('Data Augmentation')

% Rotation of image
imgflip = zeros(256,256,3,num_file);
imgflip(:,:,:,1:num_file) = rot90(img(:,:,:,:));

% Crop of image
imgcrop = zeros(256,256,3,num_file);
for i =1:num_file
    cropped = imcrop(img(:,:,:,i),[randi([0,128],1),randi([0,128],1),randi([129,256],1),randi([129,256],1)]);
    imgcrop(:,:,:,i) = imresize(cropped,[256,256]);
end

% Add noise to image
imgnoise = imnoise(img,'gaussian');

% Data combine
img = cat(4,img,imgflip,imgcrop,imgnoise);
category = [category;category;category;category];

% Random selection of data
num_file = size(img,4);
random = randperm(num_file);
img = img(:,:,:,random);
category = category(random);

%% Training with K-fold
K_fold = 10;
[indices_train, indices_test] = ML_CrossVal_KFold(K_fold, num_file);

% Feature extration, define a struct to store the features
extract_features = struct('HOG', {},'LBP',{},'HIS',{});

% HOG features extraction
disp('Processing HOG features ...')
for i = 1:num_file
    current_img = imbinarize(rgb2gray(img(:,:,:,i)));
    [HOG_data, HOGV] = extractHOGFeatures(current_img,'CellSize',[128 128]);
    extract_features{i}.HOG = HOG_data;
end 

% LBP features extraction
disp('Processing LBP features ...')
for i = 1:num_file
    current_img = imbinarize(rgb2gray(img(:,:,:,i)));
    LBP_data = extractLBPFeatures(current_img,'CellSize',[256 256],'Upright',1); %NOTE Upright = False means textures are rotationally invariant
    extract_features{i}.LBP = LBP_data;
end 

% Color Histogram features extration
disp('Processing Histogram features ...')
for i = 1:num_file
    current_img = img(:,:,:,i);
    current_img = rgb2hsv(current_img);
    HIS_data = imhist(current_img(:,:,1),30);
    HIS_data = reshape(HIS_data,1,[]);
    extract_features{i}.HIS = HIS_data;
end 
%% Training scene classification algorithm
disp('Training Classifier');

% Combine data for training
extracted_features = zeros(num_file,length(extract_features{1}.HIS)+length(extract_features{1}.HOG)+length(extract_features{1}.LBP));
for i = 1:num_file
    extracted_features(i,:) = [extract_features{i}.HIS,extract_features{i}.HOG,extract_features{i}.LBP];
end

for k = 1:K_fold

    % extract training features
    training_features = extracted_features(indices_train(k,:),:);
    training_labels = category(indices_train(k,:));
    
    % fit k-nearest neighbor classifier
    SVM = fitcknn(training_features, training_labels);

    %%  Exvaluating scene classification algorithm with each K-folds
    disp('Evaluating Classifier')

    %get the features use for test (k fold)
    test_labels = category(indices_test(k,:));
    test_features = extracted_features(indices_test(k,:),:);

    predicted_labels = predict(SVM, test_features);
    [confmat, acc(k), prec, rec, f1score] = algorithm_score(predicted_labels, test_labels)
    imagesc(confmat);
end

disp(['Average accuracy is ' num2str(sum(acc)/K_fold)]); 