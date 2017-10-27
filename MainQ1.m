%% Assignment 2
% Author: Patrick Dang Khoa Phung
% Mapping of ocean surface from stereo vision images
clear all;
clc;
clear figure;
close all;
%% Data Initialization
load('assignment2_stereodata/stereo_calib.mat');
load('assignment2_stereodata/camera_pose_data.mat');
load('assignment2_stereodata/terrain.mat');

% Plot the given terrain surface
figure
mesh(X,Y,height_grid);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Reference Terrain Data');
axis equal

% Count the number of images
num_im = length(camera_poses.left_images);

% File extraction
full_filename_1 = fullfile('assignment2_stereodata','images_left',camera_poses.left_images);
full_filename_2 = fullfile('assignment2_stereodata','images_right',camera_poses.right_images);

true_points=zeros(1400,3,49);

% Perform feature matching for each image pair
for i=1:num_im
    im_1 = imread(char(full_filename_1(i)));
    im_2 = imread(char(full_filename_2(i)));
    im_1 = rgb2gray(im_1);
    
    % Detect SURF point on each image pair 
    points1 = detectSURFFeatures(im_1,'MetricThreshold',3500);
    points2 = detectSURFFeatures(im_2,'MetricThreshold',3500);

    [descriptors1, points1] = extractFeatures(im_1, points1);
    [descriptors2, points2] = extractFeatures(im_2, points2);
    
    % Match the feature extracted from the image pair
    matched_pairs = matchFeatures(descriptors1, descriptors2);
    points1_matched = points1(matched_pairs(:, 1), :);
    points2_matched = points2(matched_pairs(:, 2), :);

    % Plot the match figure side by side
%     figure
%     showMatchedFeatures(im_1,im_2,points1_matched,points2_matched,'montage','Parent',axes);

    % Undistort the image by using the given camera parameters
    matchedPoints1 = undistortPoints(points1_matched.Location,stereoParams.CameraParameters1);
    matchedPoints2 = undistortPoints(points2_matched.Location,stereoParams.CameraParameters2);

    % Convert to world coordinate with reference to the camera 1
    worldPoints = triangulate(matchedPoints1,matchedPoints2,stereoParams);
    
    [num_point,temp]=size(worldPoints);
    
    % Now tranlate to the true position by considering rotation matrix and
    % translation matrix
    for j=1:num_point
        true_points(j,:,i) = camera_poses.R(:,:,i)\(worldPoints(j,:)' - camera_poses.t(:,i));
    end
    disp(['Doing image ', num2str(i)]);
end


% figure;
% plot3(worldPoints(:,1),worldPoints(:,3),-worldPoints(:,2),'.');
% hold on;
% caml=plotCamera('Location',[0 0 0], 'Orientation', [1,0,0; 0,0,-1; 0,1,0],'size', 0.1);
% xlabel('X(m)');
% ylabel('Y(m)');
% zlabel('Z(m)');
% axis equal

% temp=true_points(:,1);
true_points(true_points == 0) = NaN;

% Eliminate outliner for plotting
for(i=1:1400)
    for(j=1:num_im)
        if(true_points(i,3,j) < 3 || true_points(i,3,j) > 6)
           true_points(i,3,j) = NaN;
        end
    end
end

% Plot the final furface
figure;
for i=1:num_im 
    plot3(true_points(:,1,i),true_points(:,2,i),-true_points(:,3,i),'.');
    %plot3(true_points(:,1,i),true_points(:,2,i),-true_points(:,3,i),'.','markeredgecolor',z);
    hold on;
end
    
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Reconstruction using Underwater Stereo Vision');
axis equal
colormap('autumn');





