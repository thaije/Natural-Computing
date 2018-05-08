addpath src/adrem
addpath src/evaluation
addpath src/comparison_methods
addpath liblinear-2.20/matlab
addpath libsvm/matlab


% Load data
data_vgg = load_dataset('office-caltech', 'vgg');
data_surf = load_dataset('office-caltech', 'surf');


data_names = {'amazon' 'Caltech10' 'dslr' 'webcam'};
data_types = numel(data_names);

% For Ad-rem C is hardcoded to 1.0, otherwise training takes too long. See
% src/adrem/predict_liblinear_cv opts.C to enable finding the optimal C.

% For Coral, C is hardcoded to 1.0, otherwise training takes too long. See
% src/comparison_methods/predict_coral.m, line 34 to enable finding the 
% optimal C.

% loop through combinations of source / target
for s = 1:data_types
    for t = data_types:-1:1
        % skip if source == target
        if s == t 
            continue
        end
        fprintf('-----------------------------');
        fprintf('\ns=%s - t=%s\n', data_names{s}, data_names{t});
        fprintf('-----------------------------');
        
        % Ad-rem using surf
        fprintf('\nAd-rem using Surf features\n');
        [x_src, x_tgt] = preprocess(data_surf.x{s}, data_surf.y{s}, data_surf.x{t}, 'joint-std');
        y = predict_adrem(x_src, data_surf.y{1}, x_tgt);
        acc = mean(y == data_surf.y{t});
        fprintf('Final accuracy %.2f%%\n', acc);
        
        % Ad-rem using VGG
        fprintf('\nAd-rem using VGG features\n');
        [x_src, x_tgt] = preprocess(data_vgg.x{s}, data_vgg.y{s}, data_vgg.x{t}, 'joint-std');
        y = predict_adrem(x_src, data_vgg.y{1}, x_tgt);
        acc = mean(y == data_vgg.y{t});
        fprintf('Final accuracy %.2f%%\n', acc);
        
        
        % Coral using surf
        fprintf('\nCoral using Surf features\n');
        [x_src, x_tgt] = preprocess(data_surf.x{s}, data_surf.y{s}, data_surf.x{t}, 'zscore');
        y = predict_coral(x_src, data_surf.y{1}, x_tgt);
        acc = mean(y == data_surf.y{t});
        fprintf('Final accuracy %.2f%%\n', acc);
        
        % Coral using VGG
        fprintf('\nCoral using VGG features\n');
        [x_src, x_tgt] = preprocess(data_vgg.x{s}, data_vgg.y{s}, data_vgg.x{t}, 'zscore');
        y = predict_coral(x_src, data_vgg.y{1}, x_tgt);
        acc = mean(y == data_vgg.y{t});
        fprintf('Final accuracy %.2f%%\n', acc);
        
        fprintf('\n');
    end
end 

