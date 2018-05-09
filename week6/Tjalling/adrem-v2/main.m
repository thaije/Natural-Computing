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

% Arguments:
% run_methods({'dataset-name', 'features'}, labeled_data_percentage)


%%%% Exercise 1.a
% fprintf("Using vgg features with no labeled tgt data")
% result = run_methods({'office-caltech', 'vgg'}, 0.00001)
% mean_avg_accs = getfield(result, 'mean_accs')
% mean_std_accs = getfield(result, 'std_avg_accs')

%%%% Exercise 1.b
% fprintf("Using surf features with no labeled tgt data")
% result = run_methods({'office-caltech', 'surf'}, 0.00001)
% mean_avg_accs = getfield(result, 'mean_accs')
% mean_std_accs = getfield(result, 'std_avg_accs')




%%%% Exercise 3
% Do every experiment for 0.1-0.6 perc labeled tgt data, each experiment 5
% times, and plot the results

fprintf("Using surf features")
plot_methods_repeated_labeled('surf');

% fprintf("Using vgg features")
% plot_methods_repeated_labeled('vgg');





function plotted = plot_methods_repeated_labeled(features)
    labeled_data_options = [0.1 0.2 0.3 0.4 0.5 0.6];
    adrem_accs = zeros(1, numel(labeled_data_options));
    adrem_stds = zeros(1, numel(labeled_data_options));
    coral_accs = zeros(1, numel(labeled_data_options));
    coral_stds = zeros(1, numel(labeled_data_options));


    for i = 1:numel(labeled_data_options)
        labeled_data_options(i)
        result = run_methods({'office-caltech-repeated', features}, labeled_data_options(i))

        mean_avg_accs = getfield(result, 'mean_avg_accs');
        mean_std_accs = getfield(result, 'std_avg_accs'); % mean std of acc of different src-tgt sets
    %     mean_std_accs = getfield(result, 'mean_std_accs'); % mean std of avg acc between iterations

        adrem_accs(i) = mean_avg_accs(1) * 100.0;
        adrem_stds(i) = mean_std_accs(1) * 100.0;

        coral_accs(i) = mean_avg_accs(2) * 100.0;
        coral_stds(i) = mean_std_accs(2) * 100.0;
    end
    
    errorbar(labeled_data_options, adrem_accs, adrem_stds); hold on;
    errorbar(labeled_data_options, coral_accs, coral_stds);

    % show legend
    axis([0.0 0.7 0.0 100.0])
    xlabel('Percentage tgt data labeled');
    ylabel('Mean average accuracy')
    legend('Ad-rem', 'Coral');
    [lgd, icons, plots, txt] = legend('show');
    plotted = 1;
end


