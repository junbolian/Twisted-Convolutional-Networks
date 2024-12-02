%___________________________________________________________________________________________________________________________________________________%

% Twisted Convolutional Networks (TCNs) source codes (version 1.0)

% Website and codes of TCN: https://github.com/junbolian/Twisted_Convolutional_Networks

% Last update: Dec 01 2024

% E-Mail: junbolian@qq.com

% Author: Junbo Jacob Lian 

%____________________________________________________________________________________________________________________________________________________%

%% Twisted Convolutional Networks (TCNs) for Dataset with Arbitrary Features

%% Clean Environment
clear; close all; clc;

%% Set Random Seed for Reproducibility
rng(42);

%% Load Dataset
% Load data from the specified Excel file
filename = 'dataset.xlsx';
data = readmatrix(filename);

%% Shuffle Features and Separate Labels
% Assuming the last column is the label, shuffle the feature columns
labels = data(:, end);
features = data(:, 1:end-1);
num_features = size(features, 2);

% Shuffle the feature columns to increase independence between adjacent features
shuffle_idx = randperm(num_features);
features = features(:, shuffle_idx);

%% Shuffle Data Rows
idx = randperm(size(features, 1));
features = features(idx, :);
labels = labels(idx);

%% Feature Combination Configuration
% Feature Combination Layer: Create combinations of features to generate higher-order interactions
num_combinations = 2; % Number of features to combine | Options: >=2
[num_samples, num_features] = size(features);

%% Feature Combination Method Selection
% Choose the feature combination method: 'multiplicative' or 'pairwise'
combination_method = 'multiplicative'; % Options: 'multiplicative', 'pairwise'

%% Create Combinations of Features
combinations = nchoosek(1:num_features, num_combinations);
num_combined_features = size(combinations, 1);

%% Generate Combined Feature Data
combined_data = zeros(num_samples, num_combined_features);
if strcmp(combination_method, 'multiplicative')
    % Multiplicative Combination (Approach I)
    for i = 1:num_combined_features
        feature_indices = combinations(i, :);
        combined_data(:, i) = prod(features(:, feature_indices), 2); % Product as a more complex combination method
    end
elseif strcmp(combination_method, 'pairwise')
    % Summation of Pairwise Products (Approach II)
    % Calculate the sum of pairwise products for each combination
    for i = 1:num_combined_features
        feature_indices = combinations(i, :);
        pairwise_sum = 0;
        for j = 1:length(feature_indices)
            for k = (j+1):length(feature_indices)
                pairwise_sum = pairwise_sum + features(:, feature_indices(j)) .* features(:, feature_indices(k));
            end
        end
        combined_data(:, i) = pairwise_sum;
    end
end

%% Split Data into Training and Testing Sets
cv = cvpartition(labels, 'HoldOut', 0.3);
train_data = combined_data(training(cv), :);
test_data = combined_data(test(cv), :);
train_labels = labels(training(cv));
test_labels = labels(test(cv));

%% Convert Labels to Categorical
train_labels = categorical(train_labels);
test_labels = categorical(test_labels);

%% Determine Number of Classes
num_classes = numel(unique(labels));

%% TCN Model Definition
% Input Layer: Takes the combined feature set and normalizes it
layers = [...
    featureInputLayer(num_combined_features, 'Normalization', 'zscore', 'Name', 'input')
    
    % Feature Transformation Layer: Transform the combined features to capture non-linear relationships
    fullyConnectedLayer(20, 'WeightsInitializer', 'he', 'Name', 'fc1')
    batchNormalizationLayer('Name', 'batch_norm1')
    reluLayer('Name', 'relu1')
    
    % Feature Interaction Module: Further transform the features and prepare for residual connection
    fullyConnectedLayer(20, 'WeightsInitializer', 'he', 'Name', 'fc2')
    batchNormalizationLayer('Name', 'batch_norm2')
    reluLayer('Name', 'relu2')
    
    % Residual Connection: Add a residual connection to enhance gradient flow
    fullyConnectedLayer(20, 'WeightsInitializer', 'he', 'Name', 'fc3') % Added transformation layer
    batchNormalizationLayer('Name', 'batch_norm3')
    reluLayer('Name', 'relu3')
    additionLayer(2, 'Name', 'add')
    reluLayer('Name', 'relu4')
    
    % Fully Connected Layer 1: Refine the feature representation
    fullyConnectedLayer(10, 'WeightsInitializer', 'he', 'Name', 'fc4')
    
    % Dropout Layer: Prevent overfitting by randomly deactivating neurons
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    % Output Layer: Final fully connected layer followed by softmax for classification
    fullyConnectedLayer(num_classes, 'Name', 'output_fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'class_output')
];

%% Residual Connection Definition
% Define residual connection to improve gradient flow
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph, [...
    fullyConnectedLayer(20, 'WeightsInitializer', 'he', 'Name', 'skip_connection_fc')
    reluLayer('Name', 'skip_connection_relu')]); % Added ReLU activation to the skip connection
lgraph = connectLayers(lgraph, 'fc1', 'skip_connection_fc');
lgraph = connectLayers(lgraph, 'skip_connection_relu', 'add/in2');

%% Specify Training Options
% Training options for the model using Adam optimizer
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {test_data, test_labels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4);

%% Train TCN Model
% Train the TCN model with the defined architecture and options
trained_tcn = trainNetwork(train_data, train_labels, lgraph, options);

%% Evaluate TCN Model
% Evaluate the model on the test data
predicted_labels = classify(trained_tcn, test_data);
accuracy = sum(predicted_labels == test_labels) / numel(test_labels);

% Display accuracy
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Enhanced Evaluation Metrics
% Confusion Matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
figure;
confusionchart(test_labels, predicted_labels);
title('Confusion Matrix');

% Precision, Recall, and F1-score
precision = diag(conf_matrix) ./ sum(conf_matrix, 2);
recall = diag(conf_matrix) ./ sum(conf_matrix, 1)';
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;

f1_score = 2 * (precision .* recall) ./ (precision + recall);
f1_score(isnan(f1_score)) = 0;

% Display Precision, Recall, and F1-score
for i = 1:num_classes
    fprintf('Class %d - Precision: %.2f, Recall: %.2f, F1-score: %.2f\n', i, precision(i) * 100, recall(i) * 100, f1_score(i) * 100);
end

% Overall Metrics
average_precision = mean(precision);
average_recall = mean(recall);
average_f1_score = mean(f1_score);

fprintf('Average Precision: %.2f%%\n', average_precision * 100);
fprintf('Average Recall: %.2f%%\n', average_recall * 100);
fprintf('Average F1-score: %.2f%%\n', average_f1_score * 100);

%% ROC Curve and AUC
% Calculate scores for ROC
scores = predict(trained_tcn, test_data);
figure;
for i = 1:num_classes
    if any(test_labels == categorical(i)) % Check if the class exists in the test set
        [X, Y, T, AUC] = perfcurve(test_labels == categorical(i), scores(:, i), true);
        plot(X, Y);
        hold on;
        fprintf('Class %d - AUC: %.2f\n', i, AUC);
    else
        fprintf('Class %d - Not present in test data, skipping AUC calculation\n', i);
    end
end
hold off;
title('ROC Curves');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(arrayfun(@(x) sprintf('Class %d', x), find(arrayfun(@(x) any(test_labels == categorical(x)), 1:num_classes)), 'UniformOutput', false));
