import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from image_utils import load_image, normalize_img, postprocess_img_for_gif, save_image_visualization, show_image
from models import CustomConvNet
from models import CustomDeconvNet
from functools import partial
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Visualize which sections of image activate feature maps.')
    parser.add_argument('img_path', action="store", type=str,
                        help='Input image to be visualized.')
    parser.add_argument('--output_path', '-out', action="store", dest="output_path", type=str, default=None,
                        help='Path where visualization should be saved.')
    parser.add_argument('--act_percent', '-ap', action="store", dest="act_percent", default=1, 
                        help='Percentage(0.0-1.0) of activations to be visualized per feature map. Proceeds from highest to lowest activations.')
    parser.add_argument('--map_percent', '-mp', action="store", dest="map_percent", default=1,
                        help='Percentage(0.0-1.0) of feature maps to be visualized. Proceeds from highest to lowest max activations.')
    parser.add_argument('--gif', '-g', action="store_true", dest="gif",
                        help='Creates a gif for each layer that shows how the project changes as feature maps are added. Feature maps are sorted by max activation.')
    parser.add_argument('--gif_dir', '-gdir', action="store", dest="gif_dir", default="default_gifs",
                        help='Output directory to store gifs in.')
    args = parser.parse_args()

    output_path = args.img_path.split(".")[0].split("/")[-1]
    if args.output_path:
        output_path = args.output_path.split(".")[0]

    img = load_image(args.img_path)

    if args.gif:
        visualize_image_as_gif(img, output_path, args.gif_dir)
    else:
        plt.figure(num=None, figsize=(16, 12), dpi=2000)
        visualize_image(img, (float(args.act_percent), float(args.map_percent)))
        save_image_visualization(output_path)

def get_conv_nets():
    """
    Initialize a convolution net and it's corresponding
    deconvolution net using a pre-trained model to use for
    visualizations.
    """
    model_conv = CustomConvNet(models.vgg16(pretrained=True))
    model_conv.eval()
    create_storage_hooks(model_conv)
    print model_conv

    model_deconv = CustomDeconvNet(models.vgg16(pretrained=True))
    model_deconv.eval()
    print model_deconv

    return (model_conv, model_deconv)

def create_storage_hooks(model):
    """
    Make hooks to save feature maps and pooling from 
    convolutional net to be used for visualization.
    """
    def forward_hook(module, input, output, layer_id):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[layer_id] = output[0]
            model.pooling_spots[layer_id] = output[1]
        else:
            model.feature_maps[layer_id] = output

    for name, layer in enumerate(model._modules.get('features')):
        layer.register_forward_hook(partial(forward_hook, layer_id=name))

def visualize_image(img, experimental_vars):
    """
    Visualize a projection of the top feature map activations at each layer
    of the Convolutional net back to the input image using a
    Transpose Convolutional net.
    """
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    conv, deconv = get_conv_nets()
    conv(img)

    for idx, layer in enumerate(vgg16_conv_layers):
        print str.format("Layer {}", layer)        
        maps = choose_maps_by_max_activations(conv.feature_maps[layer].clone(), experimental_vars)
        deconv_output = deconv(maps, layer, conv.pooling_spots)
        output_img = normalize_img(deconv_output.data.numpy()[0])

        show_image(output_img, idx + 3, str.format('Layer {0}', layer))

def choose_maps_by_max_activations(maps, experimental_vars):
    """
    Preprocesses and selects feature maps to be visualized by prioritizing 
    feature maps with higher activation values.
        <maps> (TENSOR) Feature_maps to be preprocessed.
        <experimental_vars> (TUPLE) 
            [0] Controls how many maps will be selected per layer.
            [1] Controls how many activations will be selected per map.
        
            Values > 1 are treated as whole numbers. 
            0 > Values >= 1 are interpreted as a % of total. 
            All other values treated as 1.
    """
    map_var = experimental_vars[0]
    num_maps = maps.shape[1]
    chosen_map_count = 1
    if map_var > 1:
        chosen_map_count = map_var if num_maps > map_var else num_maps
    elif map_var > 0:
        chosen_map_count = int(map_var * num_maps)
    print str.format("{}/{} Feature Maps", chosen_map_count, num_maps)
    # Sort feature maps by max activation.
    included_feature_maps = np.array([
        torch.max(maps[0, i, :, :]).item() 
        for i in range(0, num_maps)]).argsort()[-chosen_map_count:]

    for i in range(0, num_maps):
        if i not in included_feature_maps:
            maps[:, i, :, :] = 0
        else:
            maps[0, i] = prune_feature_map_activations(
                                    maps[0, i], 
                                    experimental_vars[1])
    return maps

def visualize_image_as_gif(img, output_path, output_dir):
    """
    Visualize projections of feature maps at each layer
    of the network back to the input image using a Deconvolution
    net.
    """
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    conv, deconv = get_conv_nets()
    conv(img)

    for layer in vgg16_conv_layers:        
        num_maps = conv.feature_maps[layer].shape[1]
        sorted_feature_maps = sort_maps_by_max_activations(conv.feature_maps[layer].clone(), 1)
        gif_images = []
        
        for i in range(1, num_maps):
            print str.format("Adding {}/{} Feature Maps to gif for layer {}", i, num_maps, layer) 
            maps = conv.feature_maps[layer].clone()
            included_feature_maps = sorted_feature_maps[:i]
            for j in range(0, num_maps):
                if j not in included_feature_maps:
                    maps[:, j, :, :] = 0

            deconv_output = deconv(maps, layer, conv.pooling_spots)
            gif_images.append(postprocess_img_for_gif(deconv_output))
         
        gif_images[0].save(str.format("results/gifs/{}/{}{}.gif", output_dir, output_path, layer), save_all=True, append_images=gif_images[1:], duration=200, loop=0)


def sort_maps_by_max_activations(maps, activation_var):
    """
    Sorts feature maps to be visualized in gif from
    feature maps w/ higher activation values to lower activation values.
        <maps> (TENSOR) Feature_maps to be preprocessed.
        <activation_var> (TUPLE) Controls how many activations will be selected per map.
        
            Values > 1 are treated as whole numbers. 
            0 > Values >= 1 are interpreted as a % of total. 
            All other values treated as 1.
    """
    num_maps = maps.shape[1]
    for i in range(0, num_maps):
        maps[0, i] = prune_feature_map_activations(maps[0, i], activation_var)

    sorted_feature_maps = np.array([
        torch.max(maps[0, i, :, :]).item() 
        for i in range(0, num_maps)]).argsort()[::-1]
    return sorted_feature_maps

def prune_feature_map_activations(feature_map, activation_var):
    """
    Selects activations to be visualized.
        <feature_map> (TENSOR) Feature_maps to be preprocessed.
        <activation_var> (TUPLE) Controls how many activations will be selected per map.
        
        Values > 1 are treated as whole numbers. 
        0 > Values >= 1 are interpreted as a % of total. 
        All other values treated as 1.
    """
    nonzero_inds = torch.nonzero(feature_map[:, :])
    chosen_activation_count = 1
    if activation_var > 1:
        chosen_activation_count = activation_var if nonzero_inds > activation_var else nonzero_inds
    elif activation_var == 1:
        return feature_map
    elif activation_var > 0:
        chosen_activation_count = int(activation_var * len(nonzero_inds))

    if len(nonzero_inds) > 0:
        act_lst = []

        included_activations = torch.tensor([
            feature_map[ind[0], ind[1]] 
            for ind in nonzero_inds]).argsort()[-chosen_activation_count:]

        activation_threshold = feature_map[
                nonzero_inds[included_activations[0]][0], 
                nonzero_inds[included_activations[0]][1]]

        feature_map = torch.where(
            feature_map >= activation_threshold,
            feature_map[:, :],
            torch.zeros(feature_map[:, :].shape))

    return feature_map

if __name__ == '__main__':
    main()
    