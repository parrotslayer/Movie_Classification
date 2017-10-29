
rootFolder = 'C:\Users\Bill\Documents\GITHUB\Movie_Classification';

imds = imageDatastore(rootFolder,'IncludeSubfolders',true, 'LabelSource', 'foldernames');
