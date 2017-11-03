function [class_labels] = assign2_sceneclassifier_major(image_path)
%Where image_path is a path to an image file and class_labels is a 1 x
%N cell array of strings representing the assigned label for the scene contained
%in the image, in order of the most likely class. The strings contained in the
%array should match the If your algorithm makes a single decision of the scene
%class, then N would be equal to 1.

    img = imread(fullfile(image_path));
    load categoryClassifierMajor;

    categories = {'action','horror'};

    [labelIdx, scores] = predict(categoryClassifier, img);
    class_labels = categoryClassifier.Labels(labelIdx);
    
end

