"""
Twisted Convolutional Networks (TCNs)

Author: Junbo Jacob Lian
Email: junbolian@qq.com

Description:
This script implements the Twisted Convolutional Networks (TCNs) algorithm, which introduces a novel approach to feature combination. 
Unlike traditional Convolutional Neural Networks (CNNs), TCNs generate new feature representations through combinations of feature subsets, 
mitigating the impact of feature order. This makes TCNs particularly effective for datasets without inherent spatial or temporal relationships.

Key Features:
- Feature Combination Layer: Uses element-wise multiplication of feature pairs to create a richer feature representation.
- Robust to Feature Order: Reduces reliance on feature ordering, suitable for non-spatial datasets.
- Flexible Architecture: Includes residual connections, dropout, and batch normalization for improved training stability and model generalization.

License:
This project is licensed under the MIT License.

"""
# Twisted Convolutional Networks (TCNs) source codes (version 1.0)
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt

# Set Random Seed for Reproducibility
np.random.seed(42)


def load_dataset(filename):
    # Load data from the specified Excel file
    data = pd.read_excel(filename, engine='openpyxl')
    return data.values


# Load Dataset
filename = 'dataset.xlsx'
data = load_dataset(filename)

# Shuffle Features and Separate Labels
# Assuming the last column is the label, shuffle the feature columns
labels = data[:, -1]
features = data[:, :-1]
num_features = features.shape[1]

# Shuffle the feature columns to increase independence between adjacent features
shuffle_idx = np.random.permutation(num_features)
features = features[:, shuffle_idx]

# Shuffle Data Rows
idx = np.random.permutation(features.shape[0])
features = features[idx, :]
labels = labels[idx]

# Feature Combination Configuration
# Feature Combination Layer: Create combinations of features to generate higher-order interactions
num_combinations = 2  # Number of features to combine
num_samples, num_features = features.shape

# Feature Combination Method Selection
combination_method = 'multiplicative'  # Options: 'multiplicative', 'pairwise'

# Create Combinations of Features
combinations = list(itertools.combinations(range(num_features), num_combinations))
num_combined_features = len(combinations)

# Generate Combined Feature Data
combined_data = np.zeros((num_samples, num_combined_features))
if combination_method == 'multiplicative':
    # Multiplicative Combination (Approach I)
    for i, feature_indices in enumerate(combinations):
        combined_data[:, i] = np.prod(features[:, feature_indices], axis=1)
elif combination_method == 'pairwise':
    # Summation of Pairwise Products (Approach II)
    for i, feature_indices in enumerate(combinations):
        pairwise_sum = 0
        for j in range(len(feature_indices)):
            for k in range(j + 1, len(feature_indices)):
                pairwise_sum += features[:, feature_indices[j]] * features[:, feature_indices[k]]
        combined_data[:, i] = pairwise_sum

# Split Data into Training and Testing Sets
train_data, test_data, train_labels, test_labels = train_test_split(combined_data, labels, test_size=0.3,
                                                                    random_state=42)

# Convert Labels to Categorical
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
num_classes = train_labels.shape[1]

# TCN Model Definition
# Input Layer: Takes the combined feature set and normalizes it
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(num_combined_features,)),
    tf.keras.layers.BatchNormalization(),

    # Feature Transformation Layer
    tf.keras.layers.Dense(20, activation=None, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    # Feature Interaction Module
    tf.keras.layers.Dense(20, activation=None, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    # Residual Connection
    tf.keras.layers.Dense(20, activation=None, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Add(),
    tf.keras.layers.ReLU(),

    # Fully Connected Layer 1
    tf.keras.layers.Dense(10, activation=None, kernel_initializer='he_normal'),

    # Dropout Layer
    tf.keras.layers.Dropout(0.5),

    # Output Layer
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train TCN Model
history = model.fit(train_data, train_labels,
                    validation_data=(test_data, test_labels),
                    epochs=200,
                    batch_size=10,
                    verbose=0)

# Plot Training Progress
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy vs Epochs')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate TCN Model
predicted_labels = np.argmax(model.predict(test_data), axis=1)
test_labels_argmax = np.argmax(test_labels, axis=1)
accuracy = np.mean(predicted_labels == test_labels_argmax)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Enhanced Evaluation Metrics
conf_matrix = confusion_matrix(test_labels_argmax, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

print('Classification Report:')
print(classification_report(test_labels_argmax, predicted_labels))

# ROC Curve and AUC
plt.figure()
for i in range(num_classes):
    if np.sum(test_labels_argmax == i) > 0:
        fpr, tpr, _ = roc_curve(test_labels[:, i], model.predict(test_data)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC: {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Model Size and Resource Metrics
num_params = model.count_params()
print(f'Number of Parameters: {num_params}')
model_sizeKB = (num_params * 4) / 1024  # Assuming 4 bytes per parameter, in KB
print(f'Estimated Model Size: {model_sizeKB:.2f} KB')
