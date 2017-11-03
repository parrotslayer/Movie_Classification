%% Load up all Images


clear all
close all
clc

%% File Opening

%%
% Create an |ImageDatastore| to help you manage the data. Because
% |ImageDatastore| operates on image file locations, images are not loaded
% into memory until read, making it efficient for use with large image
% collections.
rootFolder = 'Movie_Classification';


imds = imageDatastore(rootFolder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');


%%
% The |imds| variable now contains the images and the category labels
% associated with each image. The labels are automatically assigned from
% the folder names of the image files. Use |countEachLabel| to summarize
% the number of images per category.
tbl = countEachLabel(imds);
%%
% Because |imds| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set
% is balanced.

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category;

% Use splitEachLabel method to trim the set.
imds_train = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds);

% Divide data into training and testing set
fold_ratio = 0.5;
[trainingSet, testSet] = splitEachLabel(imds, fold_ratio, 'randomize');


%% Data Pre-processing

% num_file = 1;
% 
% % Loop through each category
% for i = 1:length(classes)
%     dirs = dir(classes{i});      
%     nfiles = length(dirs);
%     % Loop through each image
%     for n=1:nfiles
%        currentfile = [root_folder, filesep, folder_dir{i}, filesep, dirs(n).name];
%        currentimage = imread(currentfile);
%        img(:,:,:,num_file) = imresize((currentimage),[1000,674]);
%        category(num_file) = i;
%        num_file = num_file + 1;
%     end
% end
% 
% num_file = num_file - 1;
% category = category';

num_file = length(imds.Files);


% Loop the image data
for i = 1:num_file
    currentimage = imds.readimage(i);
    img(:,:,:,i) = imresize((currentimage),[1000,674]);
end

category = imds.Labels;


%% Data Augmentation
% disp('Data Augmentation')

% Rotation of image
% imgflip = zeros(1000,674,3,num_file);
% imgflip(:,:,:,1:num_file) = rot90(img(:,:,:,:),2);

% Crop of image
% imgcrop = zeros(1000,674,3,num_file);
% for i =1:num_file
%     cropped = imcrop(img(:,:,:,i),[randi([0,500],1),randi([0,337],1),randi([501,674],1),randi([501,674],1)]);
%     imgcrop(:,:,:,i) = imresize(cropped,[1000,674]);
% end

% Add noise to image
% imgnoise = imnoise(img,'gaussian');

% Data combine
% img = cat(4,img,imgflip,imgcrop,imgnoise);
% category = [category;category;category;category];

% img = cat(4,img,imgcrop,imgnoise);
% category = [category;category;category];

num_file=size(img,4);

% Random selection of data
random = randperm(num_file);
img = img(:,:,:,random);
category = category(random);

%% Training with K-fold
K_fold = 10;
[indices_train, indices_test] = ML_CrossVal_KFold(K_fold, num_file);

% Feature extration, define a struct to store the features
% extract_features = struct('HOG', {},'LBP',{},'HIS',{},'SURF',{});
extract_features = struct('HOG', {},'LBP',{},'HIS',{},'SURF',{},'MSER',{},'RGB',{});

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

% Color HSV Histogram features extration
disp('Processing HSV Histogram features ...')
for i = 1:num_file
    current_img = img(:,:,:,i);
    current_img = rgb2hsv(current_img);
    HIS_data = imhist(current_img(:,:,1),1000);
    HIS_data = reshape(HIS_data,1,[]);
    extract_features{i}.HIS = HIS_data;
end 

% Color RGB Histogram features extration
disp('Processing RGB Histogram features ...')
num_bins = 10000;
for i = 1:num_file
    current_img = img(:,:,:,i);
    [counts_R,bins] = imhist(img(:,:,1,i),num_bins);
    [counts_G,bins] = imhist(img(:,:,2,i),num_bins);
    [counts_B,bins] = imhist(img(:,:,3,i),num_bins);
    extract_features{i}.RGB = [counts_R',counts_G',counts_B']; 
    
end 


% Detect SURF features extration
disp('Processing SURF features ...')
for i = 1:num_file
    current_img = imbinarize(rgb2gray(img(:,:,:,i)));
    % Detect SURF point
    points1 = detectSURFFeatures(current_img,'MetricThreshold',5000);
    [SURF_features, valid_points] = extractFeatures(current_img, points1);
    SURF_features = reshape(SURF_features,1,[]);
    SURF_features = datasample(SURF_features,20000);
    extract_features{i}.SURF = SURF_features;
end 

% Detect MSER features extration
disp('Processing MSER features ...')
for i = 1:num_file
    current_img = rgb2gray(img(:,:,:,i));
    regionsObj = detectMSERFeatures(current_img);
    % Detect SURF point
    [MSER_features, validPtsObj] = extractFeatures(current_img, regionsObj);
    MSER_features = reshape(MSER_features,1,[]);
    MSER_features = datasample(MSER_features,20000);
    extract_features{i}.MSER = MSER_features;
    
end 


%% Training scene classification algorithm
disp('Training Classifier');

% Combine data for training
extracted_features = zeros(num_file,length(extract_features{1}.HIS)+length(extract_features{1}.HOG)+length(extract_features{1}.LBP)+length(extract_features{1}.SURF)+length(extract_features{1}.MSER)+length(extract_features{1}.RGB));
for i = 1:num_file
    extracted_features(i,:) = [extract_features{i}.HIS,extract_features{i}.HOG,extract_features{i}.LBP,extract_features{i}.SURF,extract_features{i}.MSER,extract_features{i}.RGB];
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