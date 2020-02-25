import torch
import torch.nn as nn
from collections import OrderedDict

class CustomConvNet(nn.Module):
    """
    Custom convolution network architecture
    """

    def __init__(self, model):
        super(CustomConvNet, self).__init__()
        
        conv_model = self.get_conv_model(model)
        self.features = conv_model._modules["features"]

        self.feature_maps = OrderedDict()
        self.pooling_spots = OrderedDict()

    def get_conv_model(self, model):
        """
        Copy input convolutional model in a nested manner. 
        Ensure MaxPool2d block saves off indices of max activations,
        so it can be used by the corresponding MaxUnpool2d block in Deconv net.
        """
        if model._modules == OrderedDict():
            return None
        new_conv_dict = OrderedDict()
        for name, layer in model._modules.items():
            if isinstance(layer, nn.MaxPool2d):
                new_conv_dict[name] = nn.MaxPool2d(
                    layer.kernel_size, 
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    return_indices=True)
            else:
                nested_network = self.get_conv_model(layer)
                if nested_network is None:
                    new_conv_dict[name] = layer
                else:
                    new_conv_dict[name] = nested_network

        model._modules = new_conv_dict
        return model

    def forward(self, x):
        for name, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, loc = layer(x)
            else:
                x = layer(x)
        return x
