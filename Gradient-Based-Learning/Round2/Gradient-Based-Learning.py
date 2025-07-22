# Gradient-Based-Learning.py
# This file is part of the Deep Learning and Python course
import os
os.getcwd()
#os.chdir('~/Deep-Learning-and-PyTorch/Gradient-Based-Learning/Round2')
#import sys
#sys.path.append("path")
#from utils import *

import numpy as np                                
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from utils import load_styles

# load_styles()

def gradient_step_onefeature(x,y,weight,lrate):
    
    '''
    This is a function for performing gradient descent step for linear predictor and MSE as loss function.
    
    The inputs to the function are the following parameters:

     - numpy array with feature values x of shape (m,1)
     - numpy array with labels y of shape (m,1)
     - scalar value `weight`, which is the weight used for computing the prediction
     - scalar value `lrate`, which is a coefficient alpha used during weight update (learning rate)

    The function will return a new weight guess (updated weight value) and the current MSE value.   
    
    '''

    # performing the Gradient Step:
    # 1. compute predictions, given the weight vector w

    y_hat = x.dot(weight).flatten()
    
    #print('y_hat', y_hat)
    
    # 2. compute MSE loss
    error = y.flatten() - y_hat
    
    m = len(y)

    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    #print('MSE', MSE)
    
    # 3. compute the average gradient of the loss function
   
    grad_w = (-2/m)*(error.dot(x))
    
    #print('grad_w', grad_w)
    
    # 4. update the weights
    weight = (weight - (lrate * grad_w))[0]
    
    #print('weight', weight)

    return weight, MSE

#from round02 import test_gradient_step_one_feature

#test_gradient_step_one_feature(gradient_step_onefeature)

def GD_onefeature(x,y,epochs,lrate):  
    
    '''
    This is a function for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step_onefeature` performs gradient step for dataset of size `m`, 
    where each datapoint has only one feature. 
    
    The inputs to the function `GD_onefeature()` are the following parameters:
    - numpy array with the feature values x of shape (m,1)
    - numpy array with the labels y of shape (m,1)
    - scalar value `epochs`, which is the number of epochs 
    - scalar value `lrate`, which is the coefficient alpha used during weight update (learning rate)
    
    '''
    
    # initialize weight vector randomly
    np.random.seed(42)
    weight = np.random.rand()    
    # create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        # run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature(x,y,weight,lrate)
        # store current weight and training loss 
        weights.append(weight)
        loss.append(MSE)
                       
    return weights, loss

# set epoches and learning rate
epochs = 100
lrate = 0.1

# Create initial dataset for single feature examples
np.random.seed(42)
x, y = make_regression(n_samples=20, n_features=1, noise=10, random_state=42)

# store results
(weights, loss) = GD_onefeature(x,y,epochs,lrate)

# plot loss and weight values
fig, ax = plt.subplots(1,2, figsize=(10,3), sharey=True)

# Loss vs weights plot
ax[0].plot(weights, loss)
ax[0].set_xlabel("weight", fontsize=16)
ax[0].set_ylabel("Loss", fontsize=16)

# Loss vs epoch plot
ax[1].plot(range(epochs), loss)
ax[1].set_xlabel("epoch", fontsize=16)


plt.show()

# =============================================================================
# BATCH vs MINI-BATCH IMPLEMENTATION (NumPy Version)
# =============================================================================

print("\n" + "="*60)
print("BATCH vs MINI-BATCH TRAINING COMPARISON")
print("="*60)

def create_mini_batches(X, y, batch_size):
    """Create mini-batches from the dataset"""
    n_samples = X.shape[0]
    mini_batches = []
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Create mini-batches
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches

def gradient_step_batch(X, y, weight, lrate):
    """Full batch gradient descent step"""
    y_hat = X.dot(weight).flatten()
    error = y.flatten() - y_hat
    m = len(y)
    MSE = ((1.0 / m) * (np.sum(np.power(error, 2))))
    grad_w = (-2/m) * (error.dot(X))
    weight = (weight - (lrate * grad_w))[0]
    return weight, MSE

def gradient_step_mini_batch(X_batch, y_batch, weight, lrate):
    """Mini-batch gradient descent step"""
    y_hat = X_batch.dot(weight).flatten()
    error = y_batch.flatten() - y_hat
    m = len(y_batch)
    MSE = ((1.0 / m) * (np.sum(np.power(error, 2))))
    grad_w = (-2/m) * (error.dot(X_batch))
    weight = (weight - (lrate * grad_w))[0]
    return weight, MSE

def GD_full_batch(X, y, epochs, lrate):
    """Full batch gradient descent"""
    print(f"Training with FULL BATCH (dataset size: {len(X)})")
    np.random.seed(42)
    weight = np.random.rand()
    loss = []
    weights = []
    
    for epoch in range(epochs):
        weight, MSE = gradient_step_batch(X, y, weight, lrate)
        weights.append(weight)
        loss.append(MSE)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {MSE:.6f}, Weight: {weight:.6f}")
    
    return weights, loss

def GD_mini_batch(X, y, epochs, lrate, batch_size=5):
    """Mini-batch gradient descent"""
    print(f"Training with MINI-BATCH (batch size: {batch_size})")
    np.random.seed(42)
    weight = np.random.rand()
    loss = []
    weights = []
    
    for epoch in range(epochs):
        # Create mini-batches for this epoch
        mini_batches = create_mini_batches(X, y, batch_size)
        epoch_losses = []
        
        for X_batch, y_batch in mini_batches:
            weight, MSE = gradient_step_mini_batch(X_batch, y_batch, weight, lrate)
            epoch_losses.append(MSE)
        
        # Average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        weights.append(weight)
        loss.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, Weight: {weight:.6f}")
    
    return weights, loss

def GD_stochastic(X, y, epochs, lrate):
    """Stochastic gradient descent (batch_size=1)"""
    print(f"Training with SGD (batch size: 1)")
    np.random.seed(42)
    weight = np.random.rand()
    loss = []
    weights = []
    
    for epoch in range(epochs):
        epoch_losses = []
        indices = np.random.permutation(len(X))
        
        for i in indices:
            X_sample = X[i:i+1]
            y_sample = y[i:i+1]
            weight, MSE = gradient_step_mini_batch(X_sample, y_sample, weight, lrate)
            epoch_losses.append(MSE)
        
        avg_loss = np.mean(epoch_losses)
        weights.append(weight)
        loss.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, Weight: {weight:.6f}")
    
    return weights, loss

# Create a larger dataset for batch comparison
print("\nGenerating larger dataset for batch vs mini-batch comparison...")
np.random.seed(42)
X_large, y_large = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

print(f"Dataset size: {X_large.shape[0]} samples")

# Compare different training approaches
batch_epochs = 50
batch_lrate = 0.01

print("\n1. FULL BATCH TRAINING:")
weights_batch, loss_batch = GD_full_batch(X_large, y_large, batch_epochs, batch_lrate)

print("\n2. MINI-BATCH TRAINING (batch_size=10):")
weights_mini10, loss_mini10 = GD_mini_batch(X_large, y_large, batch_epochs, batch_lrate, batch_size=10)

print("\n3. MINI-BATCH TRAINING (batch_size=20):")
weights_mini20, loss_mini20 = GD_mini_batch(X_large, y_large, batch_epochs, batch_lrate, batch_size=20)

print("\n4. STOCHASTIC GRADIENT DESCENT:")
weights_sgd, loss_sgd = GD_stochastic(X_large, y_large, batch_epochs//2, batch_lrate)  # Fewer epochs

# Visualization of batch vs mini-batch comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Batch vs Mini-Batch vs SGD Comparison', fontsize=16)

# Loss convergence
axes[0, 0].plot(loss_batch, 'b-', linewidth=2, label='Full Batch')
axes[0, 0].plot(loss_mini10, 'r-', linewidth=2, label='Mini-Batch (10)')
axes[0, 0].plot(loss_mini20, 'g-', linewidth=2, label='Mini-Batch (20)')
axes[0, 0].plot(loss_sgd, 'm-', linewidth=2, label='SGD (1)', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Convergence Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Weight evolution
axes[0, 1].plot(weights_batch, 'b-', linewidth=2, label='Full Batch')
axes[0, 1].plot(weights_mini10, 'r-', linewidth=2, label='Mini-Batch (10)')
axes[0, 1].plot(weights_mini20, 'g-', linewidth=2, label='Mini-Batch (20)')
axes[0, 1].plot(weights_sgd, 'm-', linewidth=2, label='SGD (1)', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Weight Value')
axes[0, 1].set_title('Weight Evolution Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Final predictions comparison
x_test = np.linspace(X_large.min(), X_large.max(), 100).reshape(-1, 1)
y_pred_batch = x_test * weights_batch[-1]
y_pred_mini10 = x_test * weights_mini10[-1]
y_pred_mini20 = x_test * weights_mini20[-1]
y_pred_sgd = x_test * weights_sgd[-1]

axes[1, 0].scatter(X_large, y_large, alpha=0.5, color='black', s=20, label='Data')
axes[1, 0].plot(x_test, y_pred_batch, 'b-', linewidth=2, label='Full Batch')
axes[1, 0].plot(x_test, y_pred_mini10, 'r-', linewidth=2, label='Mini-Batch (10)')
axes[1, 0].plot(x_test, y_pred_mini20, 'g-', linewidth=2, label='Mini-Batch (20)')
axes[1, 0].plot(x_test, y_pred_sgd, 'm-', linewidth=2, label='SGD (1)', alpha=0.7)
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Final Model Predictions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Training efficiency analysis
batch_sizes = [1, 10, 20, len(X_large)]
final_losses = [loss_sgd[-1], loss_mini10[-1], loss_mini20[-1], loss_batch[-1]]
methods = ['SGD\n(size=1)', 'Mini-Batch\n(size=10)', 'Mini-Batch\n(size=20)', 'Full Batch\n(size=100)']

bars = axes[1, 1].bar(methods, final_losses, color=['magenta', 'red', 'green', 'blue'], alpha=0.7)
axes[1, 1].set_ylabel('Final Loss')
axes[1, 1].set_title('Final Loss Comparison')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, loss in zip(bars, final_losses):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{loss:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("BATCH vs MINI-BATCH SUMMARY (NumPy Implementation)")
print("="*60)
print("Final Results:")
print(f"Full Batch     - Final Loss: {loss_batch[-1]:.6f}, Final Weight: {weights_batch[-1]:.6f}")
print(f"Mini-Batch(10) - Final Loss: {loss_mini10[-1]:.6f}, Final Weight: {weights_mini10[-1]:.6f}")
print(f"Mini-Batch(20) - Final Loss: {loss_mini20[-1]:.6f}, Final Weight: {weights_mini20[-1]:.6f}")
print(f"SGD (size=1)   - Final Loss: {loss_sgd[-1]:.6f}, Final Weight: {weights_sgd[-1]:.6f}")

print("\nKey Observations:")
print("• Full Batch: Smoothest convergence but slower per epoch")
print("• Mini-Batch: Good balance of speed and stability")
print("• SGD: Fastest per update but noisiest convergence")
print("• All methods converge to similar final solutions")
print("="*60)

epochs = 100
lrates = [0.001, 0.01, 0.1, 0.9]

fig = plt.figure(figsize=(6,4))
weights_list = []
loss_list = []

for lrate in lrates:
    weight, loss = GD_onefeature(x, y, epochs, lrate)
    #print('lrate', lrate, 'weight', weight, 'loss', loss)
    weights_list.append(weight)
    loss_list.append(loss)

# plot results
for i,lrate in enumerate(lrates):
    plt.plot(weights_list[i], loss_list[i], label=f"lrate{lrate}")
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)  
plt.show()

fig, ax = plt.subplots(1,4, sharey=True, figsize=(15,4))

for i in range(len(lrates)):
        x_grid = np.linspace(x.min(),x.max(),100) # x-axis values for plotting
        y_pred_1 = weights_list[i][0]*x_grid # get weight computed at epoch 1
        y_pred_50 = weights_list[i][49]*x_grid # get weight computed at epoch 50
        y_pred_100 = weights_list[i][99]*x_grid # get weight computed at epoch 100
        
        ax[i].scatter(x,y) # plot data points
        ax[i].plot(x_grid, y_pred_1, label="epoch 1", c='k') # plot predictor with weight at epoch 1
        ax[i].plot(x_grid, y_pred_50, label="epoch 50", c='g') # plot predictor with weight at epoch 50
        ax[i].plot(x_grid, y_pred_100, label="epoch 100", c='r') # plot predictor with weight at epoch 100
        
        # remove top and right subplot's frames 
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        # set subplot's title
        ax[i].set_title("lrate = "+str(lrates[i]), fontsize=18)

ax[3].legend()
plt.ylim(-150,150)      
plt.show()

def gradient_step(X, y, weight, lrate):
    
    '''
    This is a function for performing gradient step with MSE loss function.
      
     The inputs:
    - numpy array (matrix) with feature values X of shape (m,n)
    - numpy array with labels y of shape (m,1)
    - numpy array `weight` of shape (n,1), which is the weight used for computing prediction
    - scalar value `lrate`, which is a coefficient alpha used during weight update

    The function will return a new weight guess (updated weight value) and current MSE value.   
    
    '''

    # performing Gradient Step:

    # 1. Compute predictions, given the feature matrix X of shape (m,n) and weight vector w of shape (n,1).
    #    Predictions should be stored in an array `y_hat` of shape (m,1).
    y_hat = (X @ weight).flatten()
    
    # 2. compute MSE loss
    
    error = y.flatten() - y_hat
    
    m = len(y)
    
    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    # 3. compute average gradient of loss function
    gradient = ((-2/m*X.T) @ error)
    
    # 4. update the weights

    for i in range(len(weight)):
        weight[i] = (weight[i] - (lrate * gradient[i]))
    
    return weight, MSE 

# from round02 import test_gradient_step

# test_gradient_step(gradient_step)

def GD(X,y,epochs,lrate):  
    '''
    This is a function for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step` performs gradient step for dataset of size `m`, 
    where each datapoint has `n` features. 
    
    '''
    
    # initialize the weight vector randomly
    np.random.seed(42)
    weight = np.random.rand(X.shape[1],1)    
    # create a list to store the loss values 
    weights = []
    loss = []
     
    for i in range(epochs):
        # run the gradient step for the whole dataset
        weight, MSE = gradient_step(X, y, weight, lrate)
        # store the MSE loss of each batch of each epoch
        weights.append(weight)
        loss.append(MSE)
                       
    return weights, loss

# generate a dataset for a regression problem and set number of features to four

X2, y2 = make_regression(n_samples=100, n_features=4, noise=20, random_state=42) 
y2 = y2.reshape(-1,1)

X2 = preprocessing.scale(X2)

# set epoch and learning rate
epochs = 100
lrate = 0.1

# store results
(weights, loss) = GD(X2, y2, epochs, lrate)

# print the cost function
fig,ax = plt.subplots(figsize=(8,4))

ax.set_ylabel('Loss', fontsize=16)
ax.set_xlabel('epochs', fontsize=16)
ax.plot(range(epochs), loss, 'b.')

plt.show()

# set epoch and learning rate
epochs = 100
lrate = 0.1

# create a linear regression model 
reg = LinearRegression(fit_intercept=False) 
# fit the linear regression model with `reg = reg.fit(data, labels)`
reg = reg.fit(x, y)
# print the optimal coefficients
print(f'Optimal weights calculated by the LinearRegression model: {reg.coef_.reshape(-1,)}')

# retrieve weight values returned by `GD_onefeature()` on the LAST epoch

# store results
(weights, loss) = GD_onefeature(x,y,epochs,lrate)
#print(weights)
gd_weight = weights[len(weights)-1]
#print('gd_weight', gd_weight)

print(f'{" "*9} Optimal weights calculated by the GD algorithm: {gd_weight.reshape(-1,)}')

import math

def GD_stop(X, y, epochs, lrate):  
    
    '''
    
    This is a function for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step_onefeature` performs gradient step for dataset of size `m`, 
    where each datapoint has `1` feature. 
    
    '''

    # initialize weight vector randomly
    np.random.seed(42)
    weight = np.random.rand()    
    # create a list to store the loss values 
    loss = []
    weights = []
    iterations = 0
    MAX_ITERATIONS = 201
    tolerance = 1e-06
    
    for i in range(epochs):
        # run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature(x,y,weight,lrate)
        # store current weight and training loss 
        weights.append(weight)
        loss.append(MSE)
        iterations += 1

        #print('iterations', iterations, 'loss[i-1]', loss[i-1], 'MSE', MSE)
        
        a = loss[i-1]
        b = MSE

        if (i > 0 and math.isclose(a,b,rel_tol=tolerance) or (iterations > MAX_ITERATIONS-1)):
            print('Stopping: ', iterations)
            break
    
    return weights, loss, iterations

# set learning rate
lrate = 0.1
epochs = 100

# store the results
(weights, loss, iterations) = GD_stop(x, y, epochs, lrate)

print(f'Number of epochs: {iterations}\nLoss: {loss[-1]}\nWeights:{weights[-1].reshape((-1, ))}')

plt.rcParams['figure.figsize'] = [20, 10]
batch_image = plt.imread("../../../coursedata/R2/batch.png")
plt.imshow(batch_image)

def batch(X,y,batch_size):
    
    '''
    This is a function for creating mini-batches of the dataset.
    The `yield` statement suspends the function’s execution and sends 
    a value back to the caller, but retains enough state to enable 
    function to resume where it is left off.  
    
    '''
    # check if the number of data points is equal in feature matrix X and label vector y
    # if the assertion fails return error message "Number of data points are different in X and y"
    assert X.shape[0] == y.shape[0], "The number of datapoints is different in X and y"
    
    # shuffle data points 
    # the permutation will randomly re-arrange the order of the numbers
    # which will be used as indices to create X and y with data points in different order
    np.random.seed(42) # for reproducibility, should NOT be used in real training
    p = np.random.permutation(len(y))
    X_perm = X[p] 
    y_perm = y[p]
    
    # generate batches
    for i in range(0,X.shape[0],batch_size):
        yield (X_perm[i:i + batch_size], y_perm[i:i + batch_size])

def minibatchSGD(X, y, batch_size, epochs, lrate):  
    
    # initialize the weight randomly
    np.random.seed(42)
    weight = np.random.rand()  
    # create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        #print('epochs', epochs)

        # Use another for-loop to iterate batch() generator and access batches one-by-one
        for mini_batch in batch(X,y,batch_size):

            X_batch, y_batch = mini_batch

            # Feed  current batch to `gradient_step_onefeature()` and get weight and loss values
            weight, MSE = gradient_step_onefeature(X_batch,y_batch,weight,lrate)

            #print('weight', weight, 'MSE',MSE)

            # Store current weight and loss values in corresponding lists
            weights.append(weight)
            
            loss.append(MSE)

            # One epoch is finished when the algorithm goes through ALL batches
  
    return weights, loss

weights, loss = minibatchSGD(x, y, 50, 2, 0.1)
print('weights', weights, 'loss', loss)

# set epoches and learning rate
epochs = 100
lrate = 0.02

# iterate through the values of `batch_sizes` param
batch_sizes = [1, 10, 100]
# list for storing weights and loss for each batch size (length of both lists=3)
# we will use these lists for plotting
weights_batches = []
loss_batches = []

for batch_size in batch_sizes:
    weights, loss = minibatchSGD(x, y, batch_size, epochs, lrate)
    weights_batches.append(weights)
    loss_batches.append(loss)
    #print('batch_size', batch_size, 'weights_batches', weights_batches, 'loss_batches', loss_batches)
    print('batch_size', batch_size)

for batch_size, weights, loss in zip(batch_sizes, weights_batches, loss_batches):
    plt.plot(weights, loss, label="batch size"+str(batch_size))
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.rcParams['figure.figsize'] = [40, 20]
plt.show()

# history of the MSE loss inccured during learning
batch_size1_loss   = loss_batches[0]
batch_size10_loss  = loss_batches[1]
batch_size100_loss = loss_batches[2]

# let's check that the length of list `loss` is equal to
# x.shape[0]/batch_size*epochs

print(f"Total number of iterations = (sample size/batch size)*epochs")
print(f"\nEpochs: {epochs}")
print(f"Sample size: {x.shape[0]}")
print(f"Batch sizes: 1, 10, 100")
print(f"Iterations per epoch: {x.shape[0]/1.0:.0f}, {x.shape[0]/10.0:.0f}, {x.shape[0]/100.0:.0f}")
print(f"Total number of iterations: {len(batch_size1_loss)}, {len(batch_size10_loss)}, {len(batch_size100_loss)}")

# display weights learnt during the SGD
print(f"\nWeights:\n\nSGD with batch size = 1 results in weight w = {weights_batches[0][-1]:.2f}\
                 \nSGD with batch size = 10 results in weight w = {weights_batches[1][-1]:.2f}\
                 \nSGD with batch size = 100 results in weight w = {weights_batches[2][-1]:.2f}")


from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(fit_intercept = False, max_iter=100, tol=1e-3)
reg.fit(x,y.reshape(-1,))

reg.coef_[0]

# create the figure and axes objects
# there will be 3 subplots in one row, the y-axis is shared between subplots
fig, axes = plt.subplots(1,3, sharey=True, figsize=(15,5))

# create lists of loss values and batch sizes for further iteration in for-loop
batch_loss_list = [batch_size1_loss, batch_size10_loss, batch_size100_loss]
batch_size      = [1,10,100] 

for ax, batch_loss, size in zip(axes, batch_loss_list, batch_size):
    # plot only first 100 values
    ax.plot(np.arange(len(batch_loss[:100])), batch_loss[:100])
    # remove top and right subplot's frames 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # set subplot's title
    ax.set_title("batch size = "+str(size), fontsize=18)

# set x- and y-axis labels
axes[0].set_xlabel('batch #', fontsize=18)
axes[0].set_ylabel('Loss', fontsize=18)

# display figure
plt.ylim(0,10000)
plt.show()

plt.rcParams['figure.figsize'] = [40, 20]

GDMomentum2_image = plt.imread("../../../coursedata/R2/GDMomentum2.gif")
plt.imshow(GDMomentum2_image)

camel3D_image = plt.imread("../../../coursedata/R2/camel3D.gif")
plt.imshow(camel3D_image)
