% Get Cropped Face from Folder 00
clear all
close all
clc

load('wiki.mat');
ind = 1;
%path_00 = zeros(600,1);
% Extract image index from Folder 00
for i = 1:length(wiki.full_path)
    path = wiki.full_path{i};
    sub_folder = path(1:2);
    % check is from the existing folders
    if strcmp('00',sub_folder)||strcmp('01',sub_folder)||strcmp('02',sub_folder) 
        % Check if there is only one face and gender can be classified
        if isnan(wiki.second_face_score(i)) && wiki.face_score(i) > 0 ...
                && not(isnan(wiki.gender(i)))
        % store index of the image from 00 folder
        path_images(ind) = i;
        ind = ind + 1;
        end
    end
end

% Get cropped image and resize
for i = 1:length(path_images)
    index = path_images(i);
    orig_img=imread(wiki.full_path{index});
    cropped_face{i}=extractSubImage(orig_img,wiki.face_location{index});
    cropped_face_resized{i} = imresize(uint8(cropped_face{i}),[256 256]);
    
%     % save the cropped image as a numbered thing
%     str=strcat(int2str(i),'.jpg');
%     % image to write
%     img = cropped_face_resized{i};
%     eval('imwrite(img,str)');   
end

save('path_0_2.mat','path_images');
save('cropped_faces_0_2.mat','cropped_face_resized');
