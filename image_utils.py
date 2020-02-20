import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def load_image(img_path):
    """
    Load image from given path and resize, normalize, and add
    original subplots.
    """
    img = Image.open(img_path)
    transform = transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        transforms.ToTensor()])
    img = transform(img)
    show_image(img.numpy(), 1, "Original Picture")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = normalize(img)
    show_image(img.numpy(), 2, "Normalized Picture")
    img = img.unsqueeze(0)
    return img

def normalize_img(img):
    """
    Normalizes numpy array.
    """
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def postprocess_img_for_gif(deconv_output):
    """
    Process img so it can be inserted into a gif.
    """
    return Image.fromarray(
                np.transpose(
                    normalize_img(deconv_output.data.numpy()[0]), 
                    (1, 2, 0)))

def save_image_visualization(path):
    """
    Save constructed matplotlib at given path.
    """
    plt.savefig(str.format("results/{}.jpeg", path))
    print(str.format('Visualization saved at {}', path))

def show_image(img, ind, title):
    """
    Add image subplot at the specified index with given title.
    """
    plt.subplot(4, 4, ind)
    plt.title(title)
    plt.imshow(np.transpose(img, (1, 2, 0)))
