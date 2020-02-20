import torch
import torch.nn as nn
from collections import OrderedDict

class CustomDeconvNet(nn.Module):
    """
    Custom transpose convolution network architecture
    """

    def __init__(self, model):
        super(CustomDeconvNet, self).__init__()

        deconv_model = self.get_deconv_model(model)
        self.features = deconv_model._modules["features"]
        self.vgg_reverse_mapping = [x for x in range(31)]
        self.alex_reverse_mapping = [x for x in range(13)]

    def get_deconv_model(self, model):
        if model._modules == OrderedDict():
            return None
        reverse_order_dict = OrderedDict()
        for name, layer in reversed(model._modules.items()):
            if isinstance(layer, nn.Conv2d):
                reverse_layer = nn.ConvTranspose2d(
                    layer.out_channels, 
                    layer.in_channels, 
                    layer.kernel_size,
                    stride=layer.stride, 
                    padding=layer.padding,
                    padding_mode=layer.padding_mode,
                    dilation=layer.dilation,
                    groups=layer.groups)
                reverse_layer.weight.data = layer.weight.data
                reverse_order_dict[name] = reverse_layer
            elif isinstance(layer, nn.MaxPool2d):
                reverse_order_dict[name] = nn.MaxUnpool2d(
                    layer.kernel_size, 
                    stride=layer.stride,
                    padding=layer.padding)
            elif isinstance(layer, nn.ReLU):
                reverse_order_dict[name]= nn.ReLU()
            elif isinstance(layer, nn.ReLU6):
                reverse_order_dict[name] = nn.ReLU6()
            elif name == "classifier":
                pass
            else:
                nested_network = self.get_deconv_model(layer)
                if nested_network is not None:
                    reverse_order_dict[name] = nested_network

        model._modules = reverse_order_dict
        return model

    def forward(self, x, layer, pool_spots):
        for idx in range(self.vgg_reverse_mapping[-layer - 1], len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_spots[self.vgg_reverse_mapping[-idx - 1]])
            else:
                x = self.features[idx](x)
        return x
