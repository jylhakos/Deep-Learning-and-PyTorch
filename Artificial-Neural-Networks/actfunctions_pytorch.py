def actfunctions_pytorch():
    '''
    Function for plotting activation functions and its gradients using PyTorch.
    '''
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec
    from matplotlib.gridspec import GridSpec
    
    # plt.style.use('seaborn')
    
    mode = input("Enter mode. Valid names are:\n'actfunc', 'gradient'\n")
    actfunc = input("Enter the name of the activation function. Valid names are:\n'relu', "
                    "'leakyrelu', 'elu', 'sigmoid', 'tanh'\n")
   
    #------DEFINE ACTIVATION FUNCTIONS USING PYTORCH----------------------#
    def relu_pytorch(z):
        """PyTorch implementation of ReLU"""
        z_tensor = torch.tensor(z, requires_grad=True)
        g_tensor = F.relu(z_tensor)
        
        # Compute gradient
        g_tensor.backward(torch.ones_like(g_tensor))
        g_grad = z_tensor.grad.numpy()
        
        return g_tensor.detach().numpy(), g_grad
    
    def leakyrelu_pytorch(z):
        """PyTorch implementation of Leaky ReLU"""
        z_tensor = torch.tensor(z, requires_grad=True)
        g_tensor = F.leaky_relu(z_tensor, negative_slope=0.1)
        
        # Compute gradient
        g_tensor.backward(torch.ones_like(g_tensor))
        g_grad = z_tensor.grad.numpy()
        
        return g_tensor.detach().numpy(), g_grad

    def elu_pytorch(z):
        """PyTorch implementation of ELU"""
        z_tensor = torch.tensor(z, requires_grad=True)
        g_tensor = F.elu(z_tensor, alpha=1.0)
        
        # Compute gradient
        g_tensor.backward(torch.ones_like(g_tensor))
        g_grad = z_tensor.grad.numpy()
        
        return g_tensor.detach().numpy(), g_grad

    def sigmoid_pytorch(z):
        """PyTorch implementation of Sigmoid"""
        z_tensor = torch.tensor(z, requires_grad=True)
        g_tensor = torch.sigmoid(z_tensor)
        
        # Compute gradient
        g_tensor.backward(torch.ones_like(g_tensor))
        g_grad = z_tensor.grad.numpy()
        
        return g_tensor.detach().numpy(), g_grad
        
    def tanh_pytorch(z):
        """PyTorch implementation of Tanh"""
        z_tensor = torch.tensor(z, requires_grad=True)
        g_tensor = torch.tanh(z_tensor)
        
        # Compute gradient
        g_tensor.backward(torch.ones_like(g_tensor))
        g_grad = z_tensor.grad.numpy()
        
        return g_tensor.detach().numpy(), g_grad

    # NumPy versions for comparison (original implementations)
    def relu(z):
        g = np.copy(z)
        g[z<0] = 0
        g_grad = np.ones(shape=g.shape)
        g_grad[z<0] = 0
        return g, g_grad
    
    def leakyrelu(z):
        alpha = 0.1
        g = np.copy(z)
        g[z<0] = alpha*z[z<0]
        g_grad = np.ones(shape=g.shape)
        g_grad[z<0] = alpha
        return g, g_grad

    def elu(z):
        alpha = 1
        g = np.copy(z)
        g[z<0] = alpha*(np.exp(z[z<0])-1)
        g_grad = np.ones(shape=g.shape)
        g_grad[z<0] = g[z<0] + alpha
        return g, g_grad

    def sigmoid(z):
        g = 1/(1+np.exp(-z))
        g_grad = g*(1-g)
        return g, g_grad
        
    def tanh(z):
        g = 2/(1+np.exp(-2*z)) - 1
        g_grad = 1-g**2
        return g, g_grad
        
    #------FUNCTION MAPPING----------------------#
    func_map = {
        'relu': relu,
        'leakyrelu': leakyrelu,
        'elu': elu,
        'sigmoid': sigmoid,
        'tanh': tanh
    }
    
    pytorch_func_map = {
        'relu': relu_pytorch,
        'leakyrelu': leakyrelu_pytorch,
        'elu': elu_pytorch,
        'sigmoid': sigmoid_pytorch,
        'tanh': tanh_pytorch
    }
    
    #------GENERATE DATA AND COMPUTE ACTIVATIONS----------------------#
    z = np.linspace(-5, 5, 100)
    
    if actfunc in func_map:
        # Compute with NumPy (original)
        g_numpy, g_grad_numpy = func_map[actfunc](z)
        
        # Compute with PyTorch
        g_pytorch, g_grad_pytorch = pytorch_func_map[actfunc](z)
        
        #------PLOTTING----------------------#
        if mode == 'actfunc':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot NumPy version
            ax1.plot(z, g_numpy, 'b-', linewidth=2, label='NumPy')
            ax1.set_title(f'{actfunc.upper()} Activation (NumPy)', fontsize=14)
            ax1.set_xlabel('Input (z)', fontsize=12)
            ax1.set_ylabel(f'{actfunc}(z)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot PyTorch version
            ax2.plot(z, g_pytorch, 'r-', linewidth=2, label='PyTorch')
            ax2.set_title(f'{actfunc.upper()} Activation (PyTorch)', fontsize=14)
            ax2.set_xlabel('Input (z)', fontsize=12)
            ax2.set_ylabel(f'{actfunc}(z)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Show comparison
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(z, g_numpy, 'b-', linewidth=2, label='NumPy', alpha=0.7)
            ax.plot(z, g_pytorch, 'r--', linewidth=2, label='PyTorch', alpha=0.7)
            ax.set_title(f'{actfunc.upper()} Activation Comparison', fontsize=14)
            ax.set_xlabel('Input (z)', fontsize=12)
            ax.set_ylabel(f'{actfunc}(z)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.show()
            
        elif mode == 'gradient':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot NumPy gradients
            ax1.plot(z, g_grad_numpy, 'b-', linewidth=2, label='NumPy')
            ax1.set_title(f'{actfunc.upper()} Gradient (NumPy)', fontsize=14)
            ax1.set_xlabel('Input (z)', fontsize=12)
            ax1.set_ylabel(f"d/dz {actfunc}(z)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot PyTorch gradients
            ax2.plot(z, g_grad_pytorch, 'r-', linewidth=2, label='PyTorch')
            ax2.set_title(f'{actfunc.upper()} Gradient (PyTorch)', fontsize=14)
            ax2.set_xlabel('Input (z)', fontsize=12)
            ax2.set_ylabel(f"d/dz {actfunc}(z)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Show comparison
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(z, g_grad_numpy, 'b-', linewidth=2, label='NumPy', alpha=0.7)
            ax.plot(z, g_grad_pytorch, 'r--', linewidth=2, label='PyTorch', alpha=0.7)
            ax.set_title(f'{actfunc.upper()} Gradient Comparison', fontsize=14)
            ax.set_xlabel('Input (z)', fontsize=12)
            ax.set_ylabel(f"d/dz {actfunc}(z)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.show()
    else:
        print(f"Unknown activation function: {actfunc}")
        print("Valid options: 'relu', 'leakyrelu', 'elu', 'sigmoid', 'tanh'")

# Class-based PyTorch activation functions for neural networks
class PyTorchActivations:
    """Collection of PyTorch activation functions for neural networks"""
    
    @staticmethod
    def create_activation(name):
        """Factory method to create activation functions"""
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(negative_slope=0.1),
            'elu': nn.ELU(alpha=1.0),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is equivalent to Swish
        }
        
        if name.lower() in activations:
            return activations[name.lower()]
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    @staticmethod
    def plot_all_activations():
        """Plot all available activation functions"""
        z = torch.linspace(-5, 5, 100)
        
        activations = ['relu', 'leakyrelu', 'elu', 'sigmoid', 'tanh']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, act_name in enumerate(activations):
            act_func = PyTorchActivations.create_activation(act_name)
            
            # Compute activation
            with torch.no_grad():
                y = act_func(z)
            
            axes[i].plot(z.numpy(), y.numpy(), linewidth=2)
            axes[i].set_title(f'{act_name.upper()}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Input')
            axes[i].set_ylabel('Output')
        
        # Hide the last subplot if not needed
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run the interactive function
    actfunctions_pytorch()
    
    # Optional: Show all activations
    print("\nShowing all activation functions:")
    PyTorchActivations.plot_all_activations()
