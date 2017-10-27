function [class_labels] = assign2_sceneclassifier(image_path)
%Where image_path is a path to an image file and class_labels is a 1 x
%N cell array of strings representing the assigned label for the scene contained
%in the image, in order of the most likely class. The strings contained in the
%array should match the If your algorithm makes a single decision of the scene
%class, then N would be equal to 1.

    img = imread(fullfile(image_path));
    load categoryClassifierFull;
    load categoryClassifierCrop;
    disp('Training');

    categories = {'ball_pit','desert','park','road','sky','snow','urban'};

    [labelIdx, scores] = predict(categoryClassifier, img);
    class_labels = categoryClassifier.Labels(labelIdx)
    
end

