% Get Cropped Face from Folder 00
ind = 1;
path_00 = zeros(600,1);
% Extract image index from Folder 00
for i = 1:length(wiki.full_path)
    path = wiki.full_path{i};
    sub_folder = path(1:2);
    if strcmp('00',sub_folder)
        % store index of the image from 00 folder
        path_00(ind) = i;
        ind = ind + 1;
    end
end

% Get cropped image and resize
for i = 1:length(path_00)
    index = path_00(i);
    orig_img=imread(wiki.full_path{index});
    cropped_face{i}=extractSubImage(orig_img,wiki.face_location{index});
    cropped_face_resized{i} = imresize(uint8(cropped_face{i}),[256 256]);
end

%% Show a cropped image i = [0,600]
close all
i = 4;
index = path_00(i);
figure
imshow(wiki.full_path{index})
figure
imshow(uint8(cropped_face{i}))

% Resize Images
B = imresize(uint8(cropped_face{i}),[256 256]);
figure
imshow(B)