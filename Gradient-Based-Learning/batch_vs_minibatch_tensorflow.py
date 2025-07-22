# Batch vs Mini-Batch Training Examples
# TensorFlow/Keras Implementation for Regular Python Files
# This file demonstrates the differences between batch and mini-batch training

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# =============================================================================
# BATCH vs MINI-BATCH THEORY
# =============================================================================

print("\n" + "="*80)
print("BATCH vs MINI-BATCH TRAINING COMPARISON")
print("="*80)

print("""
BATCH TRAINING (Batch Gradient Descent):
• Uses the entire training dataset for each gradient update
• More accurate gradient estimation
• Smoother convergence path
• Memory intensive for large datasets
• Slower per-epoch training
• Better for smaller datasets

MINI-BATCH TRAINING (Mini-Batch Gradient Descent):
• Uses small subsets of the training data for each gradient update
• Less accurate gradient estimation but faster training
• More noisy convergence path but often finds better minima
• Memory efficient for large datasets
• Faster per-epoch training
• Industry standard for deep learning

STOCHASTIC GRADIENT DESCENT (SGD):
• Uses single sample for each gradient update
• Fastest per-update but most noisy
• Can escape local minima due to noise
• Most memory efficient
""")

# =============================================================================
# DATASET GENERATION
# =============================================================================

print("\n1. DATASET GENERATION")
print("-" * 40)

# Generate regression dataset
np.random.seed(42)
X_reg, y_reg = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
X_reg = StandardScaler().fit_transform(X_reg)

# Generate classification dataset
X_clf, y_clf = make_classification(n_samples=10000, n_features=20, n_classes=2, 
                                   n_redundant=0, random_state=42)
X_clf = StandardScaler().fit_transform(X_clf)

print(f"Regression dataset shape: X={X_reg.shape}, y={y_reg.shape}")
print(f"Classification dataset shape: X={X_clf.shape}, y={y_clf.shape}")

# Split datasets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42)

# =============================================================================
# NUMPY IMPLEMENTATION
# =============================================================================

print("\n2. NUMPY IMPLEMENTATION")
print("-" * 40)

class LinearRegressionNumPy:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'batch_sizes': []}
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
    
    def forward(self, X):
        """Forward pass"""
        return X.dot(self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """Compute MSE loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def compute_gradients(self, X, y_true, y_pred):
        """Compute gradients"""
        m = X.shape[0]
        dw = (-2/m) * X.T.dot(y_true - y_pred)
        db = (-2/m) * np.sum(y_true - y_pred)
        return dw, db
    
    def update_parameters(self, dw, db):
        """Update parameters"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def train_batch(self, X, y, epochs=100):
        """Full batch training"""
        print("Training with FULL BATCH (entire dataset)...")
        self.initialize_parameters(X.shape[1])
        
        start_time = time.time()
        for epoch in range(epochs):
            # Forward pass on entire dataset
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            
            # Backward pass on entire dataset
            dw, db = self.compute_gradients(X, y, y_pred)
            self.update_parameters(dw, db)
            
            self.history['loss'].append(loss)
            self.history['batch_sizes'].append(len(X))
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}, Loss: {loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        return self.history
    
    def train_mini_batch(self, X, y, epochs=100, batch_size=32):
        """Mini-batch training"""
        print(f"Training with MINI-BATCH (batch_size={batch_size})...")
        self.initialize_parameters(X.shape[1])
        self.history = {'loss': [], 'batch_sizes': []}
        
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        start_time = time.time()
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass on mini-batch
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                
                # Backward pass on mini-batch
                dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
                self.update_parameters(dw, db)
                
                epoch_losses.append(loss)
                self.history['batch_sizes'].append(batch_size)
            
            # Average loss for the epoch
            avg_epoch_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_epoch_loss)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        return self.history
    
    def train_sgd(self, X, y, epochs=100):
        """Stochastic Gradient Descent (batch_size=1)"""
        print("Training with SGD (batch_size=1)...")
        self.initialize_parameters(X.shape[1])
        self.history = {'loss': [], 'batch_sizes': []}
        
        n_samples = X.shape[0]
        
        start_time = time.time()
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_losses = []
            
            for i in indices:
                X_sample = X[i:i+1]  # Keep 2D shape
                y_sample = y[i:i+1]
                
                # Forward pass on single sample
                y_pred = self.forward(X_sample)
                loss = self.compute_loss(y_sample, y_pred)
                
                # Backward pass on single sample
                dw, db = self.compute_gradients(X_sample, y_sample, y_pred)
                self.update_parameters(dw, db)
                
                epoch_losses.append(loss)
                self.history['batch_sizes'].append(1)
            
            avg_epoch_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_epoch_loss)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        return self.history

# Run NumPy comparison
print("\nNumPy Implementation Comparison:")
print("-" * 40)

# Full Batch
model_batch = LinearRegressionNumPy(learning_rate=0.01)
history_batch = model_batch.train_batch(X_reg_train, y_reg_train, epochs=50)

# Mini-Batch
model_mini = LinearRegressionNumPy(learning_rate=0.01)
history_mini = model_mini.train_mini_batch(X_reg_train, y_reg_train, epochs=50, batch_size=64)

# SGD
model_sgd = LinearRegressionNumPy(learning_rate=0.01)
history_sgd = model_sgd.train_sgd(X_reg_train, y_reg_train, epochs=20)  # Fewer epochs for SGD

# =============================================================================
# PANDAS IMPLEMENTATION
# =============================================================================

print("\n3. PANDAS IMPLEMENTATION")
print("-" * 40)

def create_mini_batches_pandas(df, batch_size):
    """Create mini-batches from pandas DataFrame"""
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Create batches
    batches = []
    for i in range(0, len(df_shuffled), batch_size):
        batch = df_shuffled.iloc[i:i+batch_size]
        batches.append(batch)
    
    return batches

# Convert to DataFrame
columns = [f'feature_{i}' for i in range(X_reg_train.shape[1])]
df_reg = pd.DataFrame(X_reg_train, columns=columns)
df_reg['target'] = y_reg_train

print(f"DataFrame shape: {df_reg.shape}")
print("DataFrame head:")
print(df_reg.head())

# Create different batch sizes
batch_sizes = [32, 128, 512, len(df_reg)]
batch_examples = {}

for batch_size in batch_sizes:
    if batch_size >= len(df_reg):
        batch_name = "Full Batch"
        batches = [df_reg]
    else:
        batch_name = f"Mini-Batch (size={batch_size})"
        batches = create_mini_batches_pandas(df_reg, batch_size)
    
    batch_examples[batch_name] = {
        'batch_size': batch_size,
        'num_batches': len(batches),
        'batches': batches[:2]  # Store first 2 batches as examples
    }
    
    print(f"\n{batch_name}:")
    print(f"  Number of batches: {len(batches)}")
    print(f"  Batch size: {batch_size}")
    print(f"  First batch shape: {batches[0].shape}")

# =============================================================================
# TENSORFLOW/KERAS IMPLEMENTATION
# =============================================================================

print("\n4. TENSORFLOW/KERAS IMPLEMENTATION")
print("-" * 40)

def create_keras_model(input_dim, task='regression'):
    """Create a simple Keras model"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid' if task == 'classification' else None)
    ])
    
    if task == 'regression':
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_with_different_batch_sizes(X_train, y_train, X_test, y_test, task='regression'):
    """Train models with different batch sizes"""
    batch_sizes = [16, 64, 256, len(X_train)]  # Include full batch
    results = {}
    
    for batch_size in batch_sizes:
        batch_name = "Full Batch" if batch_size >= len(X_train) else f"Batch Size {batch_size}"
        print(f"\nTraining with {batch_name}...")
        
        # Create fresh model
        model = create_keras_model(X_train.shape[1], task)
        
        # Adjust batch size for full batch
        actual_batch_size = min(batch_size, len(X_train))
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=actual_batch_size,
            epochs=20,
            validation_data=(X_test, y_test),
            verbose=0
        )
        training_time = time.time() - start_time
        
        results[batch_name] = {
            'batch_size': actual_batch_size,
            'history': history,
            'training_time': training_time,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Final loss: {history.history['loss'][-1]:.4f}")
        print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    
    return results

# Train regression models
print("Training REGRESSION models with different batch sizes:")
regression_results = train_with_different_batch_sizes(
    X_reg_train, y_reg_train, X_reg_test, y_reg_test, 'regression')

# Train classification models
print("\nTraining CLASSIFICATION models with different batch sizes:")
classification_results = train_with_different_batch_sizes(
    X_clf_train, y_clf_train, X_clf_test, y_clf_test, 'classification')

# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

print("\n5. VISUALIZATION AND ANALYSIS")
print("-" * 40)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Batch vs Mini-Batch Training Comparison', fontsize=16)

# NumPy Loss Comparison
axes[0, 0].plot(history_batch['loss'], 'b-', label='Full Batch', linewidth=2)
axes[0, 0].plot(history_mini['loss'], 'r-', label='Mini-Batch (64)', linewidth=2)
axes[0, 0].plot(history_sgd['loss'][:50], 'g-', label='SGD (size=1)', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('NumPy: Loss Convergence')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Keras Regression Loss Comparison
for name, result in regression_results.items():
    axes[0, 1].plot(result['history'].history['loss'], label=name, linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('TensorFlow/Keras: Regression Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Keras Classification Loss Comparison
for name, result in classification_results.items():
    axes[0, 2].plot(result['history'].history['loss'], label=name, linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('TensorFlow/Keras: Classification Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Training Time Comparison
batch_names = list(regression_results.keys())
reg_times = [regression_results[name]['training_time'] for name in batch_names]
clf_times = [classification_results[name]['training_time'] for name in batch_names]

x_pos = np.arange(len(batch_names))
width = 0.35

axes[1, 0].bar(x_pos - width/2, reg_times, width, label='Regression', alpha=0.7)
axes[1, 0].bar(x_pos + width/2, clf_times, width, label='Classification', alpha=0.7)
axes[1, 0].set_xlabel('Batch Configuration')
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].set_title('Training Time Comparison')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels([name.replace('Batch Size ', '') for name in batch_names], rotation=45)
axes[1, 0].legend()

# Memory Usage Simulation (Theoretical)
batch_sizes = [1, 16, 64, 256, 1024, len(X_reg_train)]
memory_usage = [b * X_reg_train.shape[1] * 4 / 1024 for b in batch_sizes]  # Approximate KB

axes[1, 1].semilogx(batch_sizes, memory_usage, 'o-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Batch Size (log scale)')
axes[1, 1].set_ylabel('Memory Usage (KB)')
axes[1, 1].set_title('Memory Usage vs Batch Size')
axes[1, 1].grid(True, alpha=0.3)

# Final Performance Comparison
final_losses_reg = [regression_results[name]['final_val_loss'] for name in batch_names]
final_losses_clf = [classification_results[name]['final_val_loss'] for name in batch_names]

axes[1, 2].bar(x_pos - width/2, final_losses_reg, width, label='Regression', alpha=0.7)
axes[1, 2].bar(x_pos + width/2, final_losses_clf, width, label='Classification', alpha=0.7)
axes[1, 2].set_xlabel('Batch Configuration')
axes[1, 2].set_ylabel('Final Validation Loss')
axes[1, 2].set_title('Final Performance Comparison')
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels([name.replace('Batch Size ', '') for name in batch_names], rotation=45)
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY AND RECOMMENDATIONS
# =============================================================================

print("\n" + "="*80)
print("BATCH vs MINI-BATCH SUMMARY")
print("="*80)

print("\nKEY FINDINGS:")
print("1. CONVERGENCE SPEED:")
print("   • Full Batch: Slower per epoch, smoother convergence")
print("   • Mini-Batch: Faster per epoch, slightly noisy but effective")
print("   • SGD: Fastest per update, very noisy but can escape local minima")

print("\n2. MEMORY EFFICIENCY:")
print("   • Full Batch: High memory usage, scales with dataset size")
print("   • Mini-Batch: Predictable memory usage, independent of dataset size")
print("   • SGD: Minimal memory usage")

print("\n3. COMPUTATIONAL EFFICIENCY:")
print("   • Full Batch: Better GPU utilization but slower overall")
print("   • Mini-Batch: Good balance of speed and stability")
print("   • SGD: Poor GPU utilization due to small batch size")

print("\nRECOMMENDED BATCH SIZES:")
print("• Small datasets (< 1000 samples): Full batch or large mini-batches")
print("• Medium datasets (1k-100k samples): Mini-batches of 32-128")
print("• Large datasets (> 100k samples): Mini-batches of 64-512")
print("• Very large datasets: Mini-batches of 256-1024")

print("\nWHEN TO USE MINI-BATCHES:")
print("✓ Large datasets that don't fit in memory")
print("✓ Need faster training iterations")
print("✓ Want regularization effect from noise")
print("✓ GPU training (better parallelization)")
print("✓ Online learning scenarios")

print("\nWHEN TO USE FULL BATCH:")
print("✓ Small datasets that fit in memory")
print("✓ Need most accurate gradient estimates")
print("✓ Convex optimization problems")
print("✓ Final fine-tuning stages")

print("="*80)
