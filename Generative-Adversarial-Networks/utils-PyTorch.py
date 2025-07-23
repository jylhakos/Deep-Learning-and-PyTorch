import torch
import torch.nn as nn

def conv_layer_visitor(conv_layer):
    """
    PyTorch version of layer visitor for convolutional layers.
    Extracts key properties from PyTorch convolutional layers.
    """
    if hasattr(conv_layer, 'out_channels'):
        filters = conv_layer.out_channels
    else:
        filters = None
        
    if hasattr(conv_layer, 'kernel_size'):
        kernel_size = conv_layer.kernel_size
    else:
        kernel_size = None
        
    if hasattr(conv_layer, 'stride'):
        strides = conv_layer.stride
    else:
        strides = None
        
    if hasattr(conv_layer, 'padding'):
        padding = conv_layer.padding
    else:
        padding = None
    
    # Determine activation function
    activation = 'linear'  # default
    if isinstance(conv_layer, nn.ReLU):
        activation = 'relu'
    elif isinstance(conv_layer, nn.LeakyReLU):
        activation = 'leaky_relu'
    elif isinstance(conv_layer, nn.Tanh):
        activation = 'tanh'
    elif isinstance(conv_layer, nn.Sigmoid):
        activation = 'sigmoid'
    
    return {
        'class_name': type(conv_layer).__name__,
        'filters': filters,
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'activation': activation
    }


def no_name_layer_visitor(layer):
    """
    PyTorch version of layer visitor for layers without special naming requirements.
    """
    layer_info = {
        'class_name': type(layer).__name__,
        'config': {}
    }
    
    # Extract common properties
    if hasattr(layer, 'in_features'):
        layer_info['config']['in_features'] = layer.in_features
    if hasattr(layer, 'out_features'):
        layer_info['config']['out_features'] = layer.out_features
    if hasattr(layer, 'num_features'):
        layer_info['config']['num_features'] = layer.num_features
    if hasattr(layer, 'p'):  # For dropout
        layer_info['config']['dropout_rate'] = layer.p
    
    return layer_info


# PyTorch layer visitors mapping
visitors = {
    'Linear': no_name_layer_visitor,
    'Conv1d': conv_layer_visitor,
    'Conv2d': conv_layer_visitor,
    'ConvTranspose2d': conv_layer_visitor,
    'MaxPool1d': no_name_layer_visitor,
    'MaxPool2d': no_name_layer_visitor,
    'AdaptiveMaxPool1d': no_name_layer_visitor,
    'Dense': no_name_layer_visitor,  # Alias for Linear
    'Dropout': no_name_layer_visitor,
    'BatchNorm1d': no_name_layer_visitor,
    'BatchNorm2d': no_name_layer_visitor,
    'Flatten': no_name_layer_visitor,
    'ReLU': no_name_layer_visitor,
    'LeakyReLU': no_name_layer_visitor,
    'Sigmoid': no_name_layer_visitor,
    'Tanh': no_name_layer_visitor
}


def layer_comparator(expected, actual):
    """
    Compare two PyTorch layers for structural similarity.
    
    Args:
        expected: Expected layer configuration (dict or PyTorch layer)
        actual: Actual PyTorch layer
        
    Returns:
        bool: True if layers match structurally, False otherwise
    """
    if isinstance(expected, dict):
        class_name = expected.get('class_name', type(actual).__name__)
    else:
        class_name = type(expected).__name__
    
    visitor = visitors.get(class_name)
    
    if visitor is None:
        # Unknown layer type, do basic comparison
        return type(expected).__name__ == type(actual).__name__
    
    if isinstance(expected, dict):
        expected_config = expected
    else:
        expected_config = visitor(expected)
    
    actual_config = visitor(actual)
    
    return expected_config == actual_config


def print_model_summary(model, input_shape=None):
    """
    Print a summary of the PyTorch model similar to Keras model.summary().
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape tuple (without batch dimension)
    """
    print(f"Model: {model.__class__.__name__}")
    print("=" * 65)
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
    print("=" * 65)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:  # Skip containers
            continue
            
        # Count parameters
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        total_params += module_params
        trainable_params += module_trainable
        
        # Get layer type
        layer_type = type(module).__name__
        
        print(f"{name:<25} {layer_type:<20} {module_params:<10}")
    
    print("=" * 65)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 65)


def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
