%clear all
clc

load('path_0_2.mat')
load('wiki.mat')
addpath(genpath('Faces_0_2_Cropped_BW'));

gender = zeros(length(path_images),1);
for i = 1:length(path_images)
    index = path_images(i);
    image_path_names{i} = strcat(int2str(i),'_BW.jpg');
    gender(i) = wiki.gender(index);
end


[InImWeight_All,e_All] = Get_Eigen_Weights(image_path_names);

%% asd

save('Eigen_Weights_All.mat','InImWeight_All','e_All')
