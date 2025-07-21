#!/usr/bin/env python3
"""
Components of Machine Learning with PyTorch

This script demonstrates the three main components of Machine Learning:

1. Data - Features and Labels
2. Hypothesis Space (Model) - PyTorch Neural Networks
3. Loss Functions - PyTorch Loss Functions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch Components of Machine Learning")
print("=====================================")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare the Iris dataset
def load_and_prepare_data():
    """Load Iris dataset and prepare for PyTorch"""
    print("\n1. DATA COMPONENT - Loading Iris Dataset")
    print("-" * 40)
    
    # Load iris data
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {iris_data.feature_names}")
    print(f"Class names: {iris_data.target_names}")
    
    # For binary classification, use only classes 1 and 2 (Versicolor vs Virginica)
    binary_mask = (y == 1) | (y == 2)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # Convert labels to 0 and 1
    label_encoder = LabelEncoder()
    y_binary = label_encoder.fit_transform(y_binary)
    
    print(f"\nBinary classification dataset shape: {X_binary.shape}")
    print(f"Binary classes distribution: {np.bincount(y_binary)}")
    
    return X, y, X_binary, y_binary, iris_data.feature_names, iris_data.target_names

# PyTorch Neural Network Models (Hypothesis Space)
class LogisticRegressionPyTorch(nn.Module):
    """PyTorch implementation of Logistic Regression"""
    
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

class MultiLayerPerceptron(nn.Module):
    """PyTorch implementation of Multi-Layer Perceptron"""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.01):
    """Train PyTorch model with validation tracking"""
    
    # Determine loss function based on model output
    if isinstance(model, LogisticRegressionPyTorch):
        criterion = nn.BCELoss()  # Binary Cross Entropy for logistic regression
    else:
        criterion = nn.CrossEntropyLoss()  # Cross Entropy for multi-class
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if isinstance(model, LogisticRegressionPyTorch):
                loss = criterion(outputs.squeeze(), batch_y.float())
                predicted = (outputs.squeeze() > 0.5).float()
            else:
                loss = criterion(outputs, batch_y.long())
                _, predicted = torch.max(outputs.data, 1)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                if isinstance(model, LogisticRegressionPyTorch):
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).float()
                else:
                    loss = criterion(outputs, batch_y.long())
                    _, predicted = torch.max(outputs.data, 1)
                
                val_loss += loss.item()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def demonstrate_binary_classification():
    """Demonstrate binary classification with PyTorch Logistic Regression"""
    print("\n2. HYPOTHESIS SPACE - PyTorch Logistic Regression")
    print("-" * 50)
    
    X, y, X_binary, y_binary, feature_names, target_names = load_and_prepare_data()
    
    # Use only first 2 features for visualization
    X_viz = X_binary[:, :2]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_viz)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create and train PyTorch logistic regression model
    model = LogisticRegressionPyTorch(input_dim=2)
    print(f"Model architecture:\n{model}")
    
    # Train model
    training_history = train_pytorch_model(model, train_loader, test_loader, num_epochs=100)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_predictions = (test_outputs.squeeze() > 0.5).cpu().numpy()
        test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Visualize results
    visualize_binary_classification(X_scaled, y_binary, model, scaler, training_history)
    
    return model, training_history

def demonstrate_multiclass_classification():
    """Demonstrate multiclass classification with PyTorch MLP"""
    print("\n3. LOSS FUNCTIONS - PyTorch Multi-Class Classification")
    print("-" * 55)
    
    X, y, _, _, feature_names, target_names = load_and_prepare_data()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create and train PyTorch MLP model
    model = MultiLayerPerceptron(input_dim=4, hidden_dim=64, num_classes=3)
    print(f"Model architecture:\n{model}")
    
    # Train model
    training_history = train_pytorch_model(model, train_loader, test_loader, num_epochs=150)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        _, test_predictions = torch.max(test_outputs, 1)
        test_predictions = test_predictions.cpu().numpy()
        test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Visualize results
    visualize_multiclass_classification(training_history, cm, target_names)
    
    return model, training_history

def visualize_binary_classification(X, y, model, scaler, history):
    """Visualize binary classification results"""
    
    # Set up plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data distribution
    sns.scatterplot(ax=axes[0, 0], x=X[:, 0], y=X[:, 1], hue=y, 
                   palette=['blue', 'red'], s=50)
    axes[0, 0].set_title('Iris Dataset: Binary Classification\n(Versicolor vs Virginica)')
    axes[0, 0].set_xlabel('Sepal Length (standardized)')
    axes[0, 0].set_ylabel('Sepal Width (standardized)')
    
    # Plot 2: Decision boundary
    model.eval()
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        mesh_tensor = torch.FloatTensor(mesh_points).to(device)
        mesh_predictions = model(mesh_tensor).cpu().numpy()
    
    mesh_predictions = mesh_predictions.reshape(xx.shape)
    
    axes[0, 1].contourf(xx, yy, mesh_predictions, alpha=0.3, levels=50, cmap='RdBu')
    sns.scatterplot(ax=axes[0, 1], x=X[:, 0], y=X[:, 1], hue=y, 
                   palette=['blue', 'red'], s=50)
    axes[0, 1].set_title('PyTorch Logistic Regression\nDecision Boundary')
    axes[0, 1].set_xlabel('Sepal Length (standardized)')
    axes[0, 1].set_ylabel('Sepal Width (standardized)')
    
    # Plot 3: Training history - Loss
    axes[1, 0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[1, 0].plot(history['val_losses'], label='Validation Loss', color='red')
    axes[1, 0].set_title('Training and Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Binary Cross-Entropy Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Training history - Accuracy
    axes[1, 1].plot(history['train_accuracies'], label='Train Accuracy', color='blue')
    axes[1, 1].plot(history['val_accuracies'], label='Validation Accuracy', color='red')
    axes[1, 1].set_title('Training and Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pytorch_binary_classification.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_multiclass_classification(history, confusion_matrix, target_names):
    """Visualize multiclass classification results"""
    
    # Set up plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training history - Loss
    axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Multi-class Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Training history - Accuracy
    axes[0, 1].plot(history['train_accuracies'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(history['val_accuracies'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Multi-class Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Confusion Matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix - PyTorch MLP')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    
    # Plot 4: Loss comparison
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    
    metrics = ['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']
    values = [final_train_loss, final_val_loss, final_train_acc/100, final_val_acc/100]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    axes[1, 1].bar(metrics, values, color=colors)
    axes[1, 1].set_title('Final Model Performance Metrics')
    axes[1, 1].set_ylabel('Value')
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pytorch_multiclass_classification.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_loss_functions():
    """Demonstrate different PyTorch loss functions"""
    print("\nPyTorch Loss Functions Demonstration")
    print("-" * 40)
    
    # Generate sample data
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred_logits = torch.tensor([[2.0, -1.0, 0.5],
                                  [-0.5, 1.5, 0.2],
                                  [0.1, -0.8, 1.2],
                                  [0.3, 0.9, -0.1],
                                  [1.8, -0.2, 0.0]])
    
    y_pred_probs = torch.softmax(y_pred_logits, dim=1)
    
    # Cross Entropy Loss
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(y_pred_logits, y_true)
    print(f"Cross Entropy Loss: {ce_value:.4f}")
    
    # Negative Log Likelihood Loss
    nll_loss = nn.NLLLoss()
    nll_value = nll_loss(torch.log(y_pred_probs), y_true)
    print(f"Negative Log Likelihood Loss: {nll_value:.4f}")
    
    # Mean Squared Error (for regression)
    y_continuous = torch.tensor([1.0, 2.5, 0.8, 1.9, 0.3])
    y_pred_continuous = torch.tensor([1.1, 2.3, 0.9, 2.1, 0.1])
    
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(y_pred_continuous, y_continuous)
    print(f"Mean Squared Error Loss: {mse_value:.4f}")
    
    # Mean Absolute Error (for regression)
    mae_loss = nn.L1Loss()
    mae_value = mae_loss(y_pred_continuous, y_continuous)
    print(f"Mean Absolute Error Loss: {mae_value:.4f}")

def main():
    """Main function to run all demonstrations"""
    print("PyTorch Components of Machine Learning Demo")
    print("=" * 50)
    
    # Load styles if available
    try:
        from utils.styles import load_styles
        load_styles()
    except ImportError:
        print("Note: Custom styles not available, using default matplotlib styles")
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    sns.set_palette("husl")
    
    # Demonstrate the three components
    print("\nDemonstrating the 3 Components of Machine Learning with PyTorch:")
    
    # Component 1: Data (already demonstrated in load_and_prepare_data)
    binary_model, binary_history = demonstrate_binary_classification()
    
    # Component 2: Hypothesis Space (Neural Network Models)
    multiclass_model, multiclass_history = demonstrate_multiclass_classification()
    
    # Component 3: Loss Functions
    demonstrate_loss_functions()
    
    print("\n" + "=" * 50)
    print("PyTorch Machine Learning Components Demo Complete!")
    print("\nKey Points:")
    print("1. DATA: Features (measurements) + Labels (categories/values)")
    print("2. HYPOTHESIS SPACE: PyTorch Neural Networks (Linear, MLP)")
    print("3. LOSS FUNCTIONS: Cross-Entropy, MSE, MAE (PyTorch implementations)")
    print("\nGenerated plots saved as PNG files for documentation.")

if __name__ == "__main__":
    main()
