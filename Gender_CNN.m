load('path_0_2.mat')
load('wiki.mat')

gender = zeros(length(path_images),1);
for i = 1:length(path_images)
    index = path_images(i);
    % get image
    %orig_img{i} = imread(wiki.full_path{index});
    % get gender
    gender(i) = wiki.gender(index);
end

%% sdfsdf
%FaceData = imageDatastore(wiki.full_path{path_images},gender);

imds = imageDatastore({'street1.jpg','street2.jpg','peppers.png','corn.tif'})
