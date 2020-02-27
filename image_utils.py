import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def load_image(img_path, single_layer=False):
    """
    Load image from given path and resize, normalize, and add
    original subplots.
    """
    img = Image.open(img_path)
    transform = transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        transforms.ToTensor()])
    orig_img = transform(img)
    if not single_layer:
        show_image(orig_img.numpy(), 1, "Original Picture")
    else:
        show_image(orig_img.numpy(), 1, "Original Picture", 1, 2)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = normalize(orig_img)
    if not single_layer:
        show_image(img.numpy(), 2, "Normalized Picture")
    img = img.unsqueeze(0)
    return img, orig_img

def normalize_img(img):
    """
    Normalizes numpy array.
    """
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def postprocess_img_for_gif(deconv_output, is_np=False):
    """
    Process img so it can be inserted into a gif.
    """
    if is_np:
        return np.transpose(
                    normalize_img(deconv_output.data.numpy()[0]), 
                    (1, 2, 0))
    else:
        return Image.fromarray(
                np.transpose(
                    normalize_img(deconv_output.data.numpy()[0]), 
                    (1, 2, 0)))



def save_image_visualization(path, layer=None):
    """
    Save constructed matplotlib at given path.
    """
    if layer:
        plt.savefig(str.format("results/images/{}{}.jpeg", path, layer))
        print(str.format('Visualization saved at {}{}', path, layer))
    else:
        plt.savefig(str.format("results/images/{}.jpeg", path))
        print(str.format('Visualization saved at {}', path))

def show_image(img, ind, title, width=4, height=4):
    """
    Add image subplot at the specified index with given title.
    """
    plt.subplot(width, height, ind)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout(pad=0.01)
    plt.imshow(np.transpose(img, (1, 2, 0)))

