# Feature-Map-in-Sights
Visualize feature map(s) activations of a trained Convolutional Network by projecting the activations back onto an input image's pixels using a DeConvolutional network.

Core implementation based on: [Zeiler M., Fergus R. Visualizing and Understanding Convolutional Networks. In ECCV 2014.](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

## VISUALIZATION MODES
#### IMAGE
>  python visualize_main.py images/tiger.jpeg -out defaults/car_vis_default -m 0.2 -a 9

![Image_Vis](results/images/all_maps_all_activation/car_vis_100_np_100_ap.jpeg)
Visualize the top 9 activations for the top %20 of feature map(s) in each layer of a CNN for the input image "images/tiger.jpeg" and save result at "defaults/tiger_vis_default".



#### GIF
>  python visualize_main.py images/tiger.jpeg --gif -out tiger

![GIF_Vis](results/gifs/tiger.gif)
Visualize projections of all activations for a set of feature map(s) from each layer to input space. GIF progresses from visualization of a single feature map to the top 64 feature maps in a layer. 

**Note:** Grid is sorted from first layer to last layer.

## EXAMPLES
Pre-generated examples for 10 sample images. Generated using the mentioned shell scripts.
- **sample images :** \images
- **sample gif_visualizations**: results\gifs
	- generate_gifs_vis.sh
- **sample image_visualizations**: results\images
	- generate_image_vis.sh

## USAGE
**img_path** ==(req)== 
Image to be visualized. Img will be resized to 224 x 224 and normalized.

> **Note**: Currently only JPEG is guaranteed support.

**-output_path** (opt): 
Relative path from FeatureMap-InSights/results where visualization(s) are stored. 

> **Note**: Don't include file extension in path, will default to JPEG or GIF.

**-gif** (opt): 
If False generates image visualization,
If True, generates gif visualization.

**IMAGE VISUALIZATION**
Determines which activations in which feature map(s) are visualized for an img. 

**-activations** (opt): Total # of activations to keep per map.

**-maps** (opt): Total # of maps to keep per layer.
* Nonzero integers represents specific count. 
* Decimal btwn 0 and 1 represents %.
* All other inputs default to a single instance. 
* Sorted by max_activation.

**Returns**: Grid containing image, normalized image, and projections for each convolutional layer. Grid is sorted from first layer to last layer.

**GIF VISUALIZATION**
Visualizes projections of feature map(s) from each layer to input space progressing from a single feature map to the top 64 feature maps in a layer. 

All activations are kept and feature map(s) are added to gif from highest to lowest max(activation) per layer.

**Returns**:  GIF of grid containing image, normalized image, and projections for each convolutional layer. Grid is sorted from first layer to last layer.

## Paper vs FeatureMap-InSights:

Topic|Paper | FeatureMap-InSights
------|----  | -------------------------
Visualization | Visualizes single feature maps for a set number of top activations per layer | Visualizes ranges of feature maps for ranges of activations together per layer.
Convolutional Net |  Krizhevksy et al |Vgg16 (In process of expanding support to  general CNN architectures)
Modes | Image | GIF and Image
## FAQ

**Why visualize ranges of feature maps and activations instead of just the top feature maps and activations?**

Since CNNs are nonlinear the superposition principle does not apply and just looking at individual feature maps separately is not enough to understand the total effects of a feature map on a network. A projection from multiple combined feature maps could produce behavior not found by just combining the projections of individual feature maps.

**Why sort feature maps by max(activation) instead of another metric?**

Max(activation) was chosen in the scenario primarily for its simplicity. Other metrics could potentially be better suited to this problem. Future iterations to this project will experiment with other metrics.

**Why is there greater noise in the projections from later layers?**

The UnPooling block in the DCNN(Deconvolutional Network) is not a perfect reversal of the pooling block. Only the indices of the max value are saved, so the other values are lost adding noise. Projections from later layers experience this effect more as they pass through more UnPooling blocks.

**Does the Deconvolution Network actually use deconvolution?**

No mathematical deconvolution is not used, the network makes use of Transpose Convolution blocks to reverse convolution blocks. However both terms are used across this project since both terms appear in the literature. 

## Future Work
- Expand supported CNNs.
- Allow for other metrics besides Max(activation) (e.g. Highest variance between activations)
- Add support for classifiers and occlusion analysis as outlined in Zeiler et al.
- Add support for more objective measures such as Mutual Information between projections and input image for different sets of feature maps.
