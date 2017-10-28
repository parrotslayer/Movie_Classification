% All Images is a Cell
function [InImWeight_All,e_All] = Get_Eigen_Weights(image_path_names)
%% load the required variables
%load('Eigen_Required_Variables.mat')

%preallocation
%[rows,cols] = size(image_path_names);
InImWeight_All = zeros(1194);
e_All = zeros(1194);

for inc = 1:1194
    disp(inc)
try
    InputImage = imread(image_path_names{inc});
catch
    disp('fuck')
end
InImage=reshape(double(InputImage)',irow*icol,1);  
temp=InImage;
me=mean(temp);
st=std(temp);
temp=(temp-me)*ustd/st+um;
NormImage = temp;
Difference = temp-m;

p = [];
aa=size(u,2);
for i = 1:aa
    pare = dot(NormImage,u(:,i));
    p = [p; pare];
end
ReshapedImage = m + u(:,1:aa)*p;    %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';

InImWeight = [];
for i=1:size(u,2)
    t = u(:,i)';
    WeightOfInputImage = dot(t,Difference');
    InImWeight = [InImWeight; WeightOfInputImage];
end

% Find Euclidean distance
e=[];
for i=1:size(omega,2)
    q = omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e = [e mag];
end

% Store in Final Array
InImWeight_All(inc,:) = rot90(InImWeight);
e_All(inc,:) = e;

end
