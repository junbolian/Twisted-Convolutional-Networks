%___________________________________________________________________________________________________________________________________________________%
%
% Twisted Convolutional Networks (TCNs) source codes (version 1.1)
% Projection-based TCN Model
% Website and codes of TCN: https://github.com/junbolian/Twisted_Convolutional_Networks
% Last update: Sep 29 2025
% E-Mail: jacoblian@u.northwestern.edu
% Author: Junbo Jacob Lian 
%____________________________________________________________________________________________________________________________________________________%

%% Twisted Convolutional Networks (TCNs) for Dataset with Arbitrary Features

%% Clean Environment
clear; close all; clc;

%% Set Random Seed for Reproducibility
rng(42);

%% Load Dataset
filename = 'dataset.xlsx';
data = readmatrix(filename);

%% Shuffle Features and Separate Labels
labels   = data(:, end);
features = data(:, 1:end-1);
[num_samples, num_features] = size(features);

% Record original feature indices for naming (for explainability)
orig_feature_idx = 1:num_features;

% Shuffle the feature columns to increase independence between adjacent features
shuffle_idx = randperm(num_features);
features = features(:, shuffle_idx);
orig_feature_idx = orig_feature_idx(shuffle_idx);  % keep mapping for names

%% Shuffle Data Rows
idx = randperm(size(features, 1));
features = features(idx, :);
labels   = labels(idx);

%% Feature Combination Configuration
num_combinations = 2; % C (>=2). Here we use pairs by default.

%% Feature Combination Method Selection
combination_method = 'pairwise'; % 'multiplicative' or 'pairwise'

%% Create Combinations of Features (choose C features each)
combinations = nchoosek(1:num_features, num_combinations);    % [num_combined_features, C]
num_combined_features = size(combinations, 1);

%% Generate Combined Feature Data  (this is z = concat{phi_S(x)})
combined_data = zeros(num_samples, num_combined_features);
switch lower(combination_method)
    case 'multiplicative' % Approach I: product of selected features
        for i = 1:num_combined_features
            idxs = combinations(i, :);
            combined_data(:, i) = prod(features(:, idxs), 2);
        end
    case 'pairwise'       % Approach II: sum of pairwise products within the subset
        for i = 1:num_combined_features
            idxs = combinations(i, :);
            s = zeros(num_samples,1);
            for j = 1:length(idxs)
                for k = (j+1):length(idxs)
                    s = s + features(:, idxs(j)) .* features(:, idxs(k));
                end
            end
            combined_data(:, i) = s;
        end
    otherwise
        error('Unknown combination_method.');
end

%% Split Data into Training and Testing Sets
cv = cvpartition(labels, 'HoldOut', 0.3);
train_data   = combined_data(training(cv), :);
test_data    = combined_data(test(cv),     :);
train_labels = labels(training(cv));
test_labels  = labels(test(cv));

%% Convert Labels to Categorical
train_labels = categorical(train_labels);
test_labels  = categorical(test_labels);

%% Determine Number of Classes
num_classes = numel(categories(categorical(labels)));

%% Hidden sizes (configurable)
H1 = 64;   % first hidden width
H2 = 256;  % second hidden / add-output width (can be smaller than num_combined_features)

%% ===== TCN Model Definition with PROJECTION-BASED residual =====
% Main branch (before addition)
mainLayers = [
    featureInputLayer(num_combined_features, 'Normalization', 'zscore', 'Name', 'input')
    
    fullyConnectedLayer(H1, 'WeightsInitializer', 'he', 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(H2, 'WeightsInitializer', 'he', 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
];

% Post-addition layers (from addition layer onwards)
postAddLayers = [
    additionLayer(2, 'Name', 'add')      % add/in1 = main; add/in2 = projected skip
    reluLayer('Name', 'post_add_relu')
    
    % Classification head
    fullyConnectedLayer(10, 'WeightsInitializer', 'he', 'Name', 'fc_head')
    dropoutLayer(0.5, 'Name', 'dropout1')
    fullyConnectedLayer(num_classes, 'Name', 'output_fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'class_output')
];

% Build graph
lgraph = layerGraph(mainLayers);
lgraph = addLayers(lgraph, postAddLayers);

% Residual source: choose 'input' (from combination layer output z) or 'relu1'
residual_source = 'input';  % 'input' is closer to "block input"; 'relu1' is also reasonable.

% Projection layers for residual connection (always include; if dims match it behaves near-identity)
projLayers = [
    fullyConnectedLayer(H2, 'WeightsInitializer', 'he', ...
        'BiasLearnRateFactor', 0, 'Name', 'proj_fc')
    batchNormalizationLayer('Name', 'proj_bn') % optional; remove if you prefer no BN
];
lgraph = addLayers(lgraph, projLayers);

% Connect paths
lgraph = connectLayers(lgraph, 'relu2', 'add/in1');      % main -> add
lgraph = connectLayers(lgraph, residual_source, 'proj_fc');
lgraph = connectLayers(lgraph, 'proj_bn', 'add/in2');    % proj skip -> add

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {test_data, test_labels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4);

%% Train TCN Model  (return training info for custom curves)
[trained_tcn, info] = trainNetwork(train_data, train_labels, lgraph, options);

%% ===== Pretty Training Curves: Loss & Accuracy (Train vs Val) =====
% Some MATLAB releases name fields slightly differently; guard with isfield.
epochs = 1:numel(info.TrainingLoss);
% Smooth curves for nicer visuals
S = 5;  % smoothing window
trLoss = smoothdata(info.TrainingLoss, 'movmean', S);
vaLoss = smoothdata(info.ValidationLoss, 'movmean', S);
trAcc  = smoothdata(info.TrainingAccuracy, 'movmean', S);
vaAcc  = smoothdata(info.ValidationAccuracy, 'movmean', S);

%% ===== Training Curves in ONE figure (Acc top, Loss bottom) =====
% Build epoch axis & pad with NaN to equal length
N = numel(info.TrainingLoss);
pad = @(v) [v(:).' nan(1, max(0, N-numel(v)))];

trLoss = pad(info.TrainingLoss);
vaLoss = pad(info.ValidationLoss);
trAcc  = pad(info.TrainingAccuracy);
vaAcc  = pad(info.ValidationAccuracy);

% Smooth (moving mean) while keeping NaNs
W = 7;  % smoothing window
trLossS = smoothdata(trLoss, 'movmean', W, 'omitnan');
vaLossS = smoothdata(vaLoss, 'movmean', W, 'omitnan');
trAccS  = smoothdata(trAcc , 'movmean', W, 'omitnan');
vaAccS  = smoothdata(vaAcc , 'movmean', W, 'omitnan');

epochs = 1:N;

% Okabe–Ito colorblind-safe palette
cAccTr = [0 114 178]/255;  % blue
cAccVa = [213 94 0]/255;   % vermillion
cLosTr = [0 158 115]/255;  % bluish green
cLosVa = [204 121 167]/255;% reddish purple

lw   = 2.2;     % line width
ms   = 5.5;     % marker size
mkEvery = max(2, round(N/12));              
mkIdx   = 1:mkEvery:N;                       

% Figure aesthetics (Nature-like)
fig = figure('Color','w','Units','pixels','Position',[100 100 760 760]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

% Helper to unify axes style
setAxesNature = @(ax) set(ax, ...
    'FontName','Helvetica', 'FontSize',11, ...
    'LineWidth',1, 'TickDir','out', ...
    'Box','off', 'XMinorTick','on', 'YMinorTick','on');

% ---- Accuracy (top) ----
ax1 = nexttile;
p1 = plot(epochs, trAccS, '-', 'Color', cAccTr, 'LineWidth', lw); hold on;
p2 = plot(epochs, vaAccS, '--', 'Color', cAccVa, 'LineWidth', lw, ...
          'Marker','o','MarkerSize',ms,'MarkerIndices',mkIdx, ...
          'MarkerFaceColor','w','MarkerEdgeColor',cAccVa);
ylabel('Accuracy (%)');
title('Training/Validation Accuracy (top) and Loss (bottom)');
grid(ax1,'on'); setAxesNature(ax1);


leg1 = legend([p1 p2], {'Train Acc','Val Acc'}, 'Location','northeast');
leg1.Box = 'off';

% ---- Loss (bottom) ----
ax2 = nexttile;
p3 = plot(epochs, trLossS, '-',  'Color', cLosTr, 'LineWidth', lw); hold on;
p4 = plot(epochs, vaLossS, '--', 'Color', cLosVa, 'LineWidth', lw, ...
          'Marker','s','MarkerSize',ms,'MarkerIndices',mkIdx, ...
          'MarkerFaceColor','w','MarkerEdgeColor',cLosVa);
xlabel('Epoch'); ylabel('Loss');
grid(ax2,'on'); setAxesNature(ax2);

leg2 = legend([p3 p4], {'Train Loss','Val Loss'}, 'Location','northeast');
leg2.Box = 'off';

xlim(ax1, [1 N]); xlim(ax2, [1 N]);

% ===== High-quality export (choose one) =====
% exportgraphics(fig, 'train_curves.pdf',  'ContentType','vector');  
% exportgraphics(fig, 'train_curves.tif',  'Resolution', 600);    
% exportgraphics(fig, 'train_curves.png',  'Resolution', 300);


%% Evaluate TCN Model
predicted_labels = classify(trained_tcn, test_data);
accuracy = mean(predicted_labels == test_labels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Enhanced Evaluation Metrics
conf_matrix = confusionmat(test_labels, predicted_labels);
figure('Color','w');
confusionchart(test_labels, predicted_labels);
title('Confusion Matrix');

precision = diag(conf_matrix) ./ sum(conf_matrix, 2);
recall    = diag(conf_matrix) ./ sum(conf_matrix, 1)';
precision(isnan(precision)) = 0;
recall(isnan(recall))       = 0;

f1_score = 2 * (precision .* recall) ./ (precision + recall);
f1_score(isnan(f1_score)) = 0;

for i = 1:num_classes
    fprintf('Class %d - Precision: %.2f, Recall: %.2f, F1-score: %.2f\n', ...
        i, precision(i) * 100, recall(i) * 100, f1_score(i) * 100);
end

avg_precision = mean(precision);
avg_recall    = mean(recall);
avg_f1        = mean(f1_score);
fprintf('Average Precision: %.2f%%\n', avg_precision * 100);
fprintf('Average Recall: %.2f%%\n',    avg_recall    * 100);
fprintf('Average F1-score: %.2f%%\n',  avg_f1        * 100);

%% ROC Curve and AUC
scores = predict(trained_tcn, test_data);
figure('Color','w'); hold on;
present = false(1, num_classes);
for i = 1:num_classes
    pres = any(test_labels == categorical(i));
    present(i) = pres;
    if pres
        [X, Y, T, AUC] = perfcurve(test_labels == categorical(i), scores(:, i), true);
        plot(X, Y, 'LineWidth', 1.8);
        fprintf('Class %d - AUC: %.2f\n', i, AUC);
    else
        fprintf('Class %d - Not present in test data, skipping AUC calculation\n', i);
    end
end
plot([0 1],[0 1],':','LineWidth',1); % diagonal reference
title('ROC Curves');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
legend(arrayfun(@(x) sprintf('Class %d', x), find(present), 'UniformOutput', false), ...
    'Location','southeastoutside');
grid on; box on; hold off;

%% ===== Explainability: Input × Gradient Attribution (no new figures) =====
% Goal: quantify each explicit interaction feature's contribution to the chosen class.

% 1) Choose class for attribution (dominant class on test set)
[~, pred_cls_idx] = max(mean(scores, 1));  % 1..num_classes
class_idx = pred_cls_idx;  % or set manually, e.g., class_idx = 1;

% 2) Compute finite-difference gradients wrt input features (test set)
num_test_samples   = size(test_data, 1);
num_combined_feats = size(test_data, 2);
gradients = zeros(num_test_samples, num_combined_feats, 'like', test_data);
epsilon   = 1e-4;

for i = 1:num_test_samples
    base_input  = test_data(i, :);
    base_scores = predict(trained_tcn, base_input);
    base_score  = base_scores(class_idx);
    for j = 1:num_combined_feats
        perturbed = base_input;
        perturbed(j) = perturbed(j) + epsilon;
        pert_scores = predict(trained_tcn, perturbed);
        gradients(i, j) = (pert_scores(class_idx) - base_score) / epsilon;
    end
end

% 3) Input×Grad attribution and normalization
input_grad_product = test_data .* gradients;
E_abs    = mean(abs(input_grad_product), 1);
E_signed = mean(input_grad_product, 1);
den = sum(E_abs) + eps;
I = E_abs / den;   % normalized attribution share per interaction feature

% 4) Human-readable names for combinations (using original indices)
comb_names = cell(num_combined_features,1);
for i = 1:num_combined_features
    idxs = combinations(i,:);          % indices in shuffled feature space
    orig_idx = orig_feature_idx(idxs); % back to original feature indices
    if num_combinations == 2
        comb_names{i} = sprintf('(%d,%d)', orig_idx(1), orig_idx(2));
    else
        comb_names{i} = sprintf('(%s)', strjoin(string(orig_idx), ','));
    end
end

% 5) Top-K list and coverage
K = min(10, num_combined_features);      % Top-10 by default
[sorted_I, order] = sort(I, 'descend');
top_idx = order(1:K);
top_cov = 100 * sum(sorted_I(1:K));

fprintf('\n==== Explainability (class %d) - Input×Grad Attribution ====\n', class_idx);
fprintf('Top-%d coverage of attribution: %.1f%% of total\n', K, top_cov);
for t = 1:K
    j = top_idx(t);
    fprintf('#%d  S=%s   importance=%.1f%%   signed_contribution=%.4f\n', ...
        t, comb_names{j}, 100*I(j), E_signed(j));
end
