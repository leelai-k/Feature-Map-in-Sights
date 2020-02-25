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
    """
    Visualize Feature Maps activations of a trained Convolutional Network
    by projecting back onto an input image's pixels using a Deconvolutional network.
    
    CLI Inputs
    INPUT_PATH/OUTPUT_PATH, 
    VISUALIZATION TYPE(gif or grid), 
    FEATURE_MAP_CONTROLS(activations and maps) 
    """
    parser = argparse.ArgumentParser(description='Visualize which sections of image activate feature maps.')
    parser.add_argument('img_path', 
                        action="store", 
                        type=str,
                        help='Color Image to be visualized\n. Currently only jpeg is guaranteed support\n.'\
                        'Img size > 224 x 224 for good results.')
    parser.add_argument('--output_path', 
                        '-out', 
                        action="store", 
                        dest="output_path", 
                        type=str, 
                        default=None,
                        help="Relative path from FeatureMap-InSights/results"\
                            "where visualization(s) are stored. Don't include file extension.")
    parser.add_argument('--layer', 
                        '-l', 
                        action="store", 
                        dest="layer", 
                        default=-1, 
                        help='Single layer to be visualized. Must be a convolutional layer.\n' \
                        '[0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]')
    parser.add_argument('--activations', 
                        '-a', 
                        action="store", 
                        dest="act_var", 
                        default=1,
                        help='Total amount of activations to choose per Feature Map.\n' \
                            'Nonzero integers specify count. Decimal btwn 0 and 1 represent percentage.\n'\
                            'Minimum of 1 activation chosen. Higher activations are prioritized')
    parser.add_argument('--maps', 
                        '-m', 
                        action="store", 
                        dest="map_var", 
                        default=1,
                        help='Total amount of Feature Maps to choose per Layer.\n' \
                            'Nonzero integers specify count. Decimal btwn 0 and 1 represent percentage.\n'\
                            'Minimum of 1 Feature Map chosen. Higher activations are prioritized')
    parser.add_argument('--show_gif', 
                        '-gif', 
                        action="store_true", 
                        dest="gif",
                        help='Creates gif where each frame adds another Feature Map to the'\
                            'visualization\n. Feature Maps are sorted by max activation.\n' \
                            'All activations are chosen for each Feature Map.')
    args = parser.parse_args()
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    output_path = args.img_path.split(".")[0].split("/")[-1]
    if args.output_path:
        output_path = args.output_path.split(".")[0]
    plt.figure(num=None, figsize=(16, 12), dpi=2000)
    img, orig_img = load_image(args.img_path, int(args.layer) in vgg16_conv_layers)

    if args.gif:
        # GIF
        visualize_image_as_gif_grid(img, output_path, orig_img)
    else:
        # GRID
        if int(args.layer) in vgg16_conv_layers:
            visualize_single_layer_in_image(img, int(args.layer), (float(args.act_var), float(args.map_var)))
            save_image_visualization(output_path, args.layer)
        else:
            visualize_image(img, (float(args.act_var), float(args.map_var)))
            save_image_visualization(output_path)

def get_conv_nets():
    """
    Initialize a convolutional net and it's corresponding
    deconvolutional net using a pre-trained model. Models share
    weights and pooling indices can be passed between them.
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
    convolutional net to be used in deconvolutional net for visualization.
    """
    def forward_hook(module, input, output, layer_id):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[layer_id] = output[0]
            model.pooling_spots[layer_id] = output[1]
        else:
            model.feature_maps[layer_id] = output

    for name, layer in enumerate(model._modules.get('features')):
        layer.register_forward_hook(partial(forward_hook, layer_id=name))

def visualize_image(img, feature_map_controls):
    """
    Visualize a projection of the top feature map activations at each layer
    of the convolutional net back to the input image using a
    transpose convolutional net.
    """
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    conv, deconv = get_conv_nets()
    conv(img)

    for idx, layer in enumerate(vgg16_conv_layers):
        print str.format("Layer {}", layer)        
        maps = choose_maps_by_max_activations(conv.feature_maps[layer].clone(), feature_map_controls)
        deconv_output = deconv(maps, layer, conv.pooling_spots)
        output_img = normalize_img(deconv_output.data.numpy()[0])

        show_image(output_img, idx + 3, str.format('Layer {0}', layer))

def visualize_single_layer_in_image(img, layer, feature_map_controls):
    """
    Visualize a projection of the top feature map activations for a given layer
    of the convolutional net back to the input image using a
    transpose convolutional net.
    """
    conv, deconv = get_conv_nets()
    conv(img)

    print str.format("Layer {}", layer)        
    maps = choose_maps_by_max_activations(conv.feature_maps[layer].clone(), feature_map_controls)
    deconv_output = deconv(maps, layer, conv.pooling_spots)
    output_img = normalize_img(deconv_output.data.numpy()[0])

    show_image(output_img, 2, str.format('Layer {0}', layer), 1, 2)

def choose_maps_by_max_activations(maps, feature_map_controls):
    """
    Preprocesses and selects feature maps to be visualized by prioritizing 
    feature maps with higher activation values.
        <maps> (TENSOR) Feature_maps to be preprocessed.
        <feature_map_controls> (TUPLE) 
            [0] Controls how many maps will be selected per layer.
            [1] Controls how many activations will be selected per map.
    """
    map_var = feature_map_controls[0]
    num_maps = maps.shape[1]
    chosen_map_count = 1
    if map_var > 1:
        chosen_map_count = int(map_var) if num_maps > map_var else num_maps
    elif map_var > 0:
        chosen_map_count = int(map_var * num_maps)
    print str.format("{}/{} Feature Maps", chosen_map_count, num_maps)

    included_feature_maps = np.array([
        torch.max(maps[0, i, :, :]).item() 
        for i in range(0, num_maps)]).argsort()[-chosen_map_count:]

    for i in range(0, num_maps):
        if i not in included_feature_maps:
            maps[:, i, :, :] = 0
        else:
            maps[0, i] = select_feature_map_activations(
                            maps[0, i], 
                            feature_map_controls[1])
    return maps

def visualize_image_as_gif(img, output_path):
    """
    Visualize projections of feature maps at each layer
    of the network back to the input image using a Deconvolutional
    net. Repeat process moving from single feature map to all feature maps.
    Generates seperate GIF for each layer.
    """
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    conv, deconv = get_conv_nets()
    conv(img)

    for layer in vgg16_conv_layers:        
        num_maps = conv.feature_maps[layer].shape[1]
        sorted_feature_maps = sort_maps_by_max_activations(conv.feature_maps[layer].clone(), 1)
        

        gif_images = []
        black_images = [postprocess_img_for_gif(torch.zeros((1, 3, 224, 224)))] * 25
        gif_images.extend(black_images)
        for i in range(1, num_maps):
            print str.format("Adding {}/{} Feature Maps to gif for layer {}", i, num_maps, layer) 
            maps = conv.feature_maps[layer].clone()
            included_feature_maps = sorted_feature_maps[:i]
            for j in range(0, num_maps):
                if j not in included_feature_maps:
                    maps[:, j, :, :] = 0

            deconv_output = deconv(maps, layer, conv.pooling_spots)
            gif_images.append(postprocess_img_for_gif(deconv_output))
        
        gif_duration_factor = 64.0 / float(num_maps)
        gif_images[0].save(
            str.format("results/gifs/{}{}.gif", output_path, layer), 
            save_all=True, 
            append_images=gif_images[1:], 
            duration=gif_duration_factor * 400, 
            loop=0)

def visualize_image_as_gif_grid(img, output_path, orig_img):
    """
    Visualize projections of feature maps at each layer
    of the network back to the input image using a Deconvolutional
    net. Repeat process moving from single feature map to all feature maps.
    Generates single gif showing layers in a grid. Layer speeds are scaled
    so that layers with many feature maps and few feature maps end together.
    """
    vgg16_conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    total_map_count = 64
    # vgg16_conv_layers = [0, 2, 5, 7, 10]
    conv, deconv = get_conv_nets()
    conv(img)
    gif_dict = {}
    # Generate scaled gif images for each layer.
    for layer in vgg16_conv_layers:        
        num_maps = conv.feature_maps[layer].shape[1]
        sorted_feature_maps = sort_maps_by_max_activations(conv.feature_maps[layer].clone(), 1)
        
        gif_scaling_factor = total_map_count / num_maps
        if gif_scaling_factor < 1:
            gif_scaling_factor = 1
        gif_images = []
        black_images = [postprocess_img_for_gif(torch.zeros((1, 3, 224, 224)))] * 2
        gif_images.extend(black_images)
        
        for i in range(1 , total_map_count + 1):
            print str.format("Adding {}/{} Feature Maps to gif for layer {}", i, num_maps, layer) 
            maps = conv.feature_maps[layer].clone()
            included_feature_maps = sorted_feature_maps[:i]
            for j in range(0, num_maps):
                if j not in included_feature_maps:
                    maps[:, j, :, :] = 0

            deconv_output = deconv(maps, layer, conv.pooling_spots)
            gif_map = [np.transpose(normalize_img(deconv_output.data.numpy()[0]), (1,2,0))] * gif_scaling_factor
            gif_images.extend(gif_map)

        gif_dict[layer] = gif_images

    grid_gif_images = []
    img = np.transpose(normalize_img(img.numpy().squeeze()), (1, 2, 0))
    orig_img = np.transpose(normalize_img(orig_img.numpy()), (1, 2, 0))

    for x in range(0, total_map_count):
        # Combine every 4 layer gifs into a row.
        grid_row_images = []
        grid_row_images.append(np.hstack([img, 
                        orig_img, 
                        gif_dict[vgg16_conv_layers[0]][x], 
                        gif_dict[vgg16_conv_layers[1]][x]]))
        row_img = []
        for layer in vgg16_conv_layers[2:]:
            row_img.append(gif_dict[layer][x])
            if len(row_img) == 4:
                grid_row_images.append(np.hstack(row_img))
                row_img = []
                
        if row_img:
            while len(row_img) < 4:
                row_img.append(normalize_img(np.zeros(np.hstack([row_img[0]]).shape)))
            grid_row_images.append(np.hstack(row_img))
        # Combine rows into grid.
        grid_gif_images.append(Image.fromarray(np.vstack(grid_row_images)))

    grid_gif_images[0].save(
        str.format("results/gifs/{}.gif", output_path), 
        save_all=True, 
        append_images=grid_gif_images[1:], 
        duration=350,
        loop=0)


def sort_maps_by_max_activations(maps, activation_control):
    """
    Sorts feature maps to be visualized in gif from
    feature maps w/ highest activation values to lowest activation values.
        <maps> (TENSOR) Feature_maps to be preprocessed.
        <activation_control> (TUPLE) Controls how many activations will be selected per map.
    """
    num_maps = maps.shape[1]
    for i in range(0, num_maps):
        maps[0, i] = select_feature_map_activations(maps[0, i], activation_control)

    sorted_feature_maps = np.array([
        torch.max(maps[0, i, :, :]).item() 
        for i in range(0, num_maps)]).argsort()[::-1]
    return sorted_feature_maps

def select_feature_map_activations(feature_map, activation_control):
    """
    Selects activations to be visualized. Higher activations are selected first.
        <feature_map> (TENSOR) Feature_maps to be preprocessed.
        <activation_control> (TUPLE) Controls how many activations will be selected per map.
    """
    nonzero_inds = torch.nonzero(feature_map[:, :])
    chosen_activation_count = 1
    if activation_control > 1:
        chosen_activation_count = int(activation_control) if nonzero_inds > activation_control else nonzero_inds
    elif activation_control == 1:
        return feature_map
    elif activation_control > 0:
        chosen_activation_count = int(activation_control * len(nonzero_inds))

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
    