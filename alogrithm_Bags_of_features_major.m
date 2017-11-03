clear all
close all
clc


img_dir = 'Movie_Classification';
action_dir = {[img_dir filesep 'Action' filesep '*.jpg']};
horror_dir = {[img_dir filesep 'Horror' filesep '*.jpg']};
romance_dir = {[img_dir filesep 'Romance' filesep '*.jpg']};
comedy_dir = {[img_dir filesep 'Comedy' filesep '*.jpg']};

classes = [action_dir,horror_dir,romance_dir,comedy_dir];
folder_dir = {'Action','Horror','Romance','Comedy'};

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
%        if (strcmp(dirs(n).name(end-7:end), 'flip.jpg') ==0)       
%             imflip(:,:,:) = rot90(currentimage);    %90 degrees
%             imwrite(imflip,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'flip.jpg'],'jpg');
%        end
%        if (strcmp(dirs(n).name(end-7:end), 'crop.jpg') ==0) 
%            imcropped = imcrop(currentimage,[randi([0,128],1),randi([0,128],1),randi([129,256],1),randi([129,256],1)]);
%            imcropped = imresize(imcropped,[256,256]);
%            imwrite(imcropped,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'crop.jpg'],'jpg');
%        end
%        if (strcmp(dirs(n).name(end-7:end), 'nois.jpg') ==0)       
%             imgnoise = imnoise(currentimage,'gaussian');    %90 degrees
%             imwrite(imgnoise,[img_dir, filesep, folder_dir{i}, filesep, dirs(n).name(1:end-4), 'nois.jpg'],'jpg');
%        end

    end
end


disp('Training');

rootFolder = fullfile(pwd,'Movie_Classification');
categories = {'action','horror'};

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
[trainingSet, validationSet] = partition(imgSet, 0.7, 'randomize')


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
%     for n=1:nfiles
%        currentfile = [img_dir, filesep, folder_dir{i}, filesep, dirs(n).name];
%        currentimage = imread(currentfile);
%        if (strcmp(dirs(n).name(end-7:end), 'flip.jpg') ~=0)       
%             delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
%        end
%        if (strcmp(dirs(n).name(end-7:end), 'crop.jpg') ~=0)       
%             delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
%        end
%        if (strcmp(dirs(n).name(end-7:end), 'nois.jpg') ~=0)       
%             delete ([img_dir, filesep, folder_dir{i}, filesep, dirs(n).name]);
%        end
%    end
end