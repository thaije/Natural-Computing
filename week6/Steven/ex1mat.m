%% Load data sets
% Load VGG data
VGG.amazon = load('office-vgg-sumpool-amazon-fc6'); % Let's call this the source data
VGG.dslr = load('office-vgg-sumpool-dslr-fc6'); % Arbitrary target data

% Load SURF data
SURF.caltech = load('Caltech10_SURF_L10'); % Let's call this the source data
SURF.amazon = load('amazon_SURF_L10.mat'); % Arbitrary target data

%% Preprocessing 
% SURF.amazon.fts = prepro(SURF.amazon.fts); % Advised on source github
% SURF.caltech.fts = prepro(SURF.caltech.fts);

% VGG.amazon.x = prepro(VGG.amazon.x);
% SURF.dslr.x = prepro(VGG.dslr.x);

%% CORAL
% On SURF data set
SURF.caltech.ftsCORAL = CORAL(SURF.caltech.fts, SURF.amazon.fts);

% On VGG data set
VGG.amazon.xCORAL = CORAL(VGG.amazon.x, VGG.dslr.x);

