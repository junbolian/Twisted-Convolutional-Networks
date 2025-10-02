"""
Twisted Convolutional Networks (TCNs): Enhancing Feature Interactions for Non-Spatial Data Classification
Projection-based TCN Model
Original MATLAB code converted to Python using TensorFlow/Keras
Author: Junbo Jacob Lian
Date: Sep 29, 2025
https://github.com/junbolian/Twisted-Convolutional-Networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from itertools import combinations as comb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TCNClassifier:
    def __init__(self, num_combinations=2, combination_method='pairwise', H1=64, H2=256):
        """
        Initialize TCN Classifier
        
        Parameters:
        num_combinations: int, number of features to combine (C >= 2)
        combination_method: str, 'multiplicative' or 'pairwise'
        H1: int, first hidden layer width
        H2: int, second hidden layer/add-output width
        """
        self.num_combinations = num_combinations
        self.combination_method = combination_method
        self.H1 = H1
        self.H2 = H2
        self.model = None
        self.scaler = StandardScaler()
        self.feature_combinations = None
        self.orig_feature_idx = None
        
    def load_and_prepare_data(self, filename):
        """Load dataset from Excel file and prepare features/labels"""
        # Load data
        data = pd.read_excel(filename, header=None).values
        
        # Separate features and labels
        self.labels = data[:, -1].astype(int)
        features = data[:, :-1]
        num_samples, num_features = features.shape
        
        # Record original feature indices for explainability
        self.orig_feature_idx = np.arange(num_features)
        
        # Shuffle feature columns to increase independence
        shuffle_idx = np.random.permutation(num_features)
        features = features[:, shuffle_idx]
        self.orig_feature_idx = self.orig_feature_idx[shuffle_idx]
        
        # Shuffle data rows
        idx = np.random.permutation(num_samples)
        self.features = features[idx]
        self.labels = self.labels[idx]
        
        return self.features, self.labels
    
    def create_feature_combinations(self):
        """Create combinations of features"""
        num_features = self.features.shape[1]
        self.feature_combinations = list(comb(range(num_features), self.num_combinations))
        num_combined_features = len(self.feature_combinations)
        
        # Generate combined feature data
        num_samples = self.features.shape[0]
        combined_data = np.zeros((num_samples, num_combined_features))
        
        if self.combination_method.lower() == 'multiplicative':
            # Approach I: product of selected features
            for i, idxs in enumerate(self.feature_combinations):
                combined_data[:, i] = np.prod(self.features[:, idxs], axis=1)
        elif self.combination_method.lower() == 'pairwise':
            # Approach II: sum of pairwise products within the subset
            for i, idxs in enumerate(self.feature_combinations):
                s = np.zeros(num_samples)
                for j in range(len(idxs)):
                    for k in range(j + 1, len(idxs)):
                        s += self.features[:, idxs[j]] * self.features[:, idxs[k]]
                combined_data[:, i] = s
        else:
            raise ValueError(f"Unknown combination_method: {self.combination_method}")
        
        return combined_data
    
    def build_tcn_model(self, input_dim, num_classes):
        """Build TCN model with projection-based residual connection"""
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        # Normalize input
        normalized = layers.BatchNormalization(name='input_norm')(input_layer)
        
        # Main branch
        x = layers.Dense(self.H1, kernel_initializer='he_normal', name='fc1')(normalized)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.ReLU(name='relu1')(x)
        
        x = layers.Dense(self.H2, kernel_initializer='he_normal', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.ReLU(name='relu2')(x)
        
        # Projection branch for residual connection
        # Project input to match H2 dimensions
        proj = layers.Dense(self.H2, kernel_initializer='he_normal', 
                           use_bias=False, name='proj_fc')(normalized)
        proj = layers.BatchNormalization(name='proj_bn')(proj)
        
        # Addition layer (residual connection)
        added = layers.Add(name='add')([x, proj])
        added = layers.ReLU(name='post_add_relu')(added)
        
        # Classification head
        x = layers.Dense(10, kernel_initializer='he_normal', name='fc_head')(added)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(num_classes, name='output_fc')(x)
        output = layers.Softmax(name='softmax')(x)
        
        # Create model
        model = models.Model(inputs=input_layer, outputs=output, name='TCN')
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32, lr=1e-3):
        """Train the TCN model"""
        # Get number of classes
        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Build model
        self.model = self.build_tcn_model(X_train.shape[1], num_classes)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=30, 
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Normalize test data
        X_test_norm = self.scaler.transform(X_test)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_norm)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Print metrics for each class
        num_classes = len(np.unique(y_test))
        for i in range(num_classes):
            print(f'Class {i} - Precision: {precision[i]*100:.2f}, '
                  f'Recall: {recall[i]*100:.2f}, F1-score: {f1[i]*100:.2f}')
        
        # Average metrics
        print(f'Average Precision: {np.mean(precision)*100:.2f}%')
        print(f'Average Recall: {np.mean(recall)*100:.2f}%')
        print(f'Average F1-score: {np.mean(f1)*100:.2f}%')
        
        return y_pred, y_pred_proba
    
    def plot_training_curves(self, history):
        """Plot training and validation curves"""
        # Smooth function
        def smooth_curve(points, factor=0.7):
            smoothed = []
            for point in points:
                if smoothed:
                    previous = smoothed[-1]
                    smoothed.append(previous * factor + point * (1 - factor))
                else:
                    smoothed.append(point)
            return smoothed
        
        # Get data
        epochs = range(1, len(history.history['loss']) + 1)
        train_loss = smooth_curve(history.history['loss'])
        val_loss = smooth_curve(history.history['val_loss'])
        train_acc = smooth_curve(history.history['accuracy'])
        val_acc = smooth_curve(history.history['val_accuracy'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Accuracy subplot
        ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, val_acc, 'r--', label='Validation Accuracy', 
                linewidth=2, marker='o', markevery=10, markersize=5)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training/Validation Accuracy and Loss')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Loss subplot
        ax2.plot(epochs, train_loss, 'g-', label='Train Loss', linewidth=2)
        ax2.plot(epochs, val_loss, 'm--', label='Validation Loss', 
                linewidth=2, marker='s', markevery=10, markersize=5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
        
    def plot_roc_curves(self, y_test, y_pred_proba):
        """Plot ROC curves for each class"""
        # Binarize labels for multi-class ROC
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            if np.any(y_test_bin[:, i]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'Class {i} (AUC = {roc_auc:.2f})')
                print(f'Class {i} - AUC: {roc_auc:.2f}')
            else:
                print(f'Class {i} - Not present in test data, skipping AUC calculation')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def explain_features(self, X_test, y_test, class_idx=None):
        """
        Feature importance using gradient-based attribution
        """
        X_test_norm = self.scaler.transform(X_test)
        
        if class_idx is None:
            # Use dominant class
            y_pred_proba = self.model.predict(X_test_norm)
            class_idx = np.argmax(np.mean(y_pred_proba, axis=0))
        
        # Convert to TensorFlow tensors
        X_tensor = tf.convert_to_tensor(X_test_norm, dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor, training=False)
            class_output = predictions[:, class_idx]
        
        gradients = tape.gradient(class_output, X_tensor)
        
        # Input × Gradient attribution
        input_grad_product = X_test_norm * gradients.numpy()
        E_abs = np.mean(np.abs(input_grad_product), axis=0)
        E_signed = np.mean(input_grad_product, axis=0)
        
        # Normalize
        I = E_abs / (np.sum(E_abs) + 1e-10)
        
        # Create feature names
        comb_names = []
        for i, idxs in enumerate(self.feature_combinations):
            orig_idx = self.orig_feature_idx[list(idxs)]
            if self.num_combinations == 2:
                comb_names.append(f'({orig_idx[0]},{orig_idx[1]})')
            else:
                comb_names.append(f'({",".join(map(str, orig_idx))})')
        
        # Top-K features
        K = min(10, len(comb_names))
        top_idx = np.argsort(I)[::-1][:K]
        top_cov = 100 * np.sum(I[top_idx])
        
        print(f'\n==== Explainability (class {class_idx}) - Input×Grad Attribution ====')
        print(f'Top-{K} coverage of attribution: {top_cov:.1f}% of total')
        
        for t, j in enumerate(top_idx):
            print(f'#{t+1}  S={comb_names[j]}   importance={100*I[j]:.1f}%   '
                  f'signed_contribution={E_signed[j]:.4f}')


def main():
    """Main execution function"""
    # Initialize TCN
    tcn = TCNClassifier(
        num_combinations=2,
        combination_method='pairwise',
        H1=64,
        H2=256
    )
    
    # Load and prepare data
    filename = 'dataset.xlsx'  # Change this to your file path
    features, labels = tcn.load_and_prepare_data(filename)
    
    # Create feature combinations
    combined_data = tcn.create_feature_combinations()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train model
    print("Training TCN model...")
    history = tcn.train(X_train, y_train, X_test, y_test, epochs=200)
    
    # Plot training curves
    tcn.plot_training_curves(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_pred_proba = tcn.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    tcn.plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curves
    tcn.plot_roc_curves(y_test, y_pred_proba)
    
    # Feature explanation
    tcn.explain_features(X_test, y_test)
    
    # Save model (optional)
    # tcn.model.save('tcn_model.h5')
    print("\nTraining complete!")


if __name__ == "__main__":

    main()
