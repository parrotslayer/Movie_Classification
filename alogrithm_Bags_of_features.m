clear all
close all
clc


img_dir = 'dataset';
ballpit_dir = {[img_dir filesep 'ball_pit' filesep '*.jpg']};
desert_dir = {[img_dir filesep 'desert' filesep '*.jpg']};
park_dir = {[img_dir filesep 'park' filesep '*.jpg']};
road_dir = {[img_dir filesep 'road' filesep '*.jpg']};
sky_dir = {[img_dir filesep 'sky' filesep '*.jpg']};
snow_dir = {[img_dir filesep 'snow' filesep '*.jpg']};
urban_dir = {[img_dir filesep 'urban' filesep '*.jpg']};

classes = [ballpit_dir,desert_dir,park_dir,road_dir,sky_dir,snow_dir,urban_dir];
folder_dir = {'ball_pit','desert','park','road','sky','snow','urban'};

num_file = 1;

disp('Data Augmentation');

% loop through each categories
for i = 1:length(classes)
    dirs = dir(classes{i});      
    nfiles = length(dirs);    % Number of files found
    % loop each image in the category
    for n=1:nfiles
       currentfile = [img_dir, filesep, folder_dir{i}, filesep, dirs(n).name];
       currentimage = imread(currentfile);
       if (strcmp(dirs(n).name(end-7:end), 'flip.jpg') ==0)       
            imflip(:,:,:) = rot90(currentimage);    %90 degrees
            imwrite(imflip,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'flip.jpg'],'jpg');
       end
       if (strcmp(dirs(n).name(end-7:end), 'crop.jpg') ==0) 
           imcropped = imcrop(currentimage,[randi([0,128],1),randi([0,128],1),randi([129,256],1),randi([129,256],1)]);
           imcropped = imresize(imcropped,[256,256]);
           imwrite(imcropped,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'crop.jpg'],'jpg');
       end
       if (strcmp(dirs(n).name(end-7:end), 'nois.jpg') ==0)       
            imgnoise = imnoise(currentimage,'gaussian');    %90 degrees
            imwrite(imgnoise,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'nois.jpg'],'jpg');
       end

    end
end


disp('Training');

rootFolder = fullfile(pwd,'dataset');
categories = {'ball_pit','desert','park','road','sky','snow','urban'};

imds = imageDatastore(rootFolder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
% imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
% countEachLabel(imds)
% 
% [trainingSet, validationSet] = splitEachLabel(imds, 0.1, 'randomize');

imgSet=imageSet(rootFolder,'recursive');
[trainingSet, validationSet] = partition(imgSet, 0.9, 'randomize')


bag = bagOfFeatures(trainingSet);


img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')


categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

confMatrix = evaluate(categoryClassifier, validationSet);
imagesc(confMatrix);

%% Delete the generated images for data augmentation
% loop through each categories
for i = 1:length(classes)
    dirs = dir(classes{i});      
    nfiles = length(dirs);    % Number of files found
    % loop each image in the category
    for n=1:nfiles
       currentfile = [img_dir, filesep, folder_dir{i}, filesep, dirs(n).name];
       currentimage = imread(currentfile);
       if (strcmp(dirs(n).name(end-7:end), 'flip.jpg') ~=0)       
            delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
       end
       if (strcmp(dirs(n).name(end-7:end), 'crop.jpg') ~=0)       
            delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
       end
       if (strcmp(dirs(n).name(end-7:end), 'nois.jpg') ~=0)       
            delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
       end
    end
end