close all
clear all
clc

%% Define variables
% root folder
rootFolder = 'C:\Users\Bill\Documents\GITHUB\Movie_Classification\Database';
% make IMDS
imds = imageDatastore(rootFolder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');
        
% load the generated SVM Eigen Face Weight model
load('Mdl_0_2_weights.mat')

% If no faces will default to NaN
Pred_Gender_CNN = NaN(length(imds.Files),2);
Pred_Gender_Eigen = NaN(length(imds.Files),2);

% Create a detector object.
faceDetector = vision.CascadeObjectDetector;

%length(imds.Files)
for inc = 1:2
    
    % Read input image.
    I = imread(imds.Files{inc});
    
    % Convert to Greyscale
    I_BW = rgb2gray(I);
    
    % Detect faces.
    bboxes = step(faceDetector, I);
    
    % Crop faces and resize
    clear rows cols
    [rows,cols] = size(bboxes);
    %% Only do gender recognition if there are faces
    if rows > 0
        % Crop out the face detected
        for i = 1:rows
            Movie_Faces_Eigen{i} = I_BW(bboxes(i,2):bboxes(i,2)+bboxes(i,4),bboxes(i,1):bboxes(i,1)+bboxes(i,3));
            % Has to be resized into the size of the faces used for training
            Movie_Faces_Eigen{i} = imresize(Movie_Faces_Eigen{i},[256 256]);
        end
        
        % Compute the eigen faces
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
                label_show{j} = 'M';
            else
                Females_Eigen = Females_Eigen + 1;
                label_show{j} = 'F';

            end
        end
        
        % Annotate detected faces.
        IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, label_show);
        % Do not plot it
        f = figure('visible','off'); 
        imshow(IFaces)
        title('Gender Recognition Using Eigen Faces')
        
        % Save the annotated poster
        string(inc,:) = strcat(num2str(inc),'_Annotated_Eigen','.jpg');
        saveas(gcf,string(inc,:));
       
        Pred_Gender_CNN(inc,:) = [Females_Eigen,Males_Eigen];
        %Pred_Gender_Eigen = NaN(length(imds.Files,2));

        %extract faces
        
        %convert to required format(s)
        % so 64x64x3 for CNN
        % 256x256 BW for eigen
        
        %Store 64 into imds
        
        %run extract eigen face weights
        %run SVM to get genders
        
        %run CNN onto the new faces
        
    end %end if boxxes > 0 loop
    
    %store both into array
end