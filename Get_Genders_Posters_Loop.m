close all
clear all
clc

%% Define variables
% root folder
rootFolder = 'C:\Users\Bill\Documents\GITHUB\Movie_Classification\Database';
% make IMDS
imds = imageDatastore(rootFolder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');

%% path for save fig
addpath(genpath('altmany-export_fig-9ac0917'))

% Load convnet
load('convnet_100epoch.mat')
% load the generated SVM Eigen Face Weight model
load('Mdl_0_2_weights.mat')
% load required variables for comuting eigen faces
load('Eigen_Required_Variables.mat')
global um;
global ustd;
global u;
global irow;
global icol;
global temp;
global m;
global M;
global omega;

% If no faces will default to NaN
Pred_Gender_CNN = NaN(length(imds.Files),2);
Pred_Gender_Eigen = NaN(length(imds.Files),2);

% Create a detector object.
faceDetector = vision.CascadeObjectDetector;

%length(imds.Files)
for inc = 1:length(imds.Files)
    disp(inc)
    % Read input image.
    I = imread(imds.Files{inc});
    
    % Convert to Greyscale
    I_BW = rgb2gray(I);
    
    % Detect faces.
    clear bboxes
    bboxes = step(faceDetector, I);
    
    % Crop faces and resize
    clear rows cols
    [rows,cols] = size(bboxes);
    %% Only do gender recognition if there are faces
    if rows > 0
        % Crop out the face detected
        for i = 1:rows
            Movie_Faces{i} = I_BW(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3));
            % Has to be resized into the size of the faces used for training
            Movie_Faces_Eigen{i} = imresize(Movie_Faces{i},[256 256]);
            Movie_Faces_CNN{i} = imresize(Movie_Faces{i},[64 64]);
            % Save movie face
            str = strcat('CNN_face_',int2str(i),'.jpg');
            file_names{i} = str;
            % image to write
            imwrite(Movie_Faces_CNN{i},str);
        end
        
        %% Classify the Images in the Test Data and Compute Accuracy
        %Store images into IMDS
        labels = ones(rows,1)*inc;
        FaceData = imageDatastore(file_names);
        
        % Run the trained network on the test set that was not used to train the
        % network and predict the image labels (digits).
        label_pred_cat = classify(convnet,FaceData);
        
        % Convert catagorical to cell
        clear label_CNN
        Males_CNN = 0;
        Females_CNN = 0;
        for i = 1:rows
            if label_pred_cat(i) == categorical(0)
                label_CNN{i} = 'Female';
                Females_CNN = Females_CNN + 1;
            else
                label_CNN{i} = 'Male';
                Males_CNN = Males_CNN + 1;
            end
        end
        
        %% Classify using Eigen Faces' Weights
        [Weight_All,e_All] = Get_Eigen_Weights_RawData(Movie_Faces_Eigen);
        
        % Apply ML Model Using Omega
        featureVector2 = [Weight_All,e_All];
        
        % Pass features into predict. Returns vector with predicted
        label_pred = predict(Mdl,featureVector2);
        
        %% Get number of Males and Females
        Males_Eigen = 0;
        Females_Eigen = 0;
        clear label_show
        for j = 1:rows
            if label_pred(j) == 1
                Males_Eigen = Males_Eigen + 1;
                label_show{j} = 'Male';
            else
                Females_Eigen = Females_Eigen + 1;
                label_show{j} = 'Female';               
            end
        end
        
        %% Annotate detected faces from Eigen.
        IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_show);
        % Do not plot it
        f = figure('visible','off');
        imshow(IFaces)
        title('Gender Recognition Using Eigen Faces')
        
        % Save the annotated poster        
        export_fig(sprintf('%d_Annotated_Eigen.jpg', inc),'-native') 

        %% Annotate detected faces from CNN
        IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_CNN);
        % Do not plot it
        f = figure('visible','off');
        imshow(IFaces)
        title('Gender Recognition Using a CNN')
        
        % Save the annotated poster        
        export_fig(sprintf('%d_Annotated_CNN.jpg', inc),'-native') 
        
        % Store predicted genders into arrays
        Pred_Gender_CNN(inc,:) = [Females_Eigen,Males_Eigen];
        Pred_Gender_Eigen(inc,:) = [Females_CNN,Males_CNN];
        
    end %end if boxxes > 0 loop
    
end