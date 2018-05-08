addpath src/adrem
addpath src/evaluation
addpath src/comparison_methods
addpath liblinear-2.20/matlab
addpath libsvm/matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% First install libsvm and liblinear as described in Readme.md in /Tjalling
% Then run this file 

% Files: 
% - Coral is in /src/comparison_methods/predict_coral.m
% - Adrem is in /src/adrem/predict_adrem.m
% - reading in data is in /src/evaluation/load_dataset.m
% - pipeline which calls all that stuff is in /src/evaluation/run_methods.m


% Changes: 
% - Ad-rem C parameter is hardcoded to 1.0, to minimize training time. See
%   src/adrem/predict_liblinear_cv opts.C to enable finding the optimal C.
% - For Coral C  hardcoded to 1.0 to minimize training time. 
%   See src/comparison_methods/predict_coral.m, line 34 to enable finding the 
%   optimal C.
% - Preprocessing is set to 'joint-std' to minimize training time. 
%   See line 180 % of /src/evaluation/run_methods.m to change it back 



fprintf("Using surf features")
result = run_methods({'office-caltech', 'surf'})

fprintf("Using vgg features")
result = run_methods({'office-caltech', 'vgg'})

% Do the experiment 5 times (with 'office-caltech-repeated' option)
% fprintf("Using surf features")
% result = run_methods({'office-caltech-repeated', 'surf'})

% fprintf("Using vgg features")
% result = run_methods({'office-caltech-repeated', 'vgg'})
