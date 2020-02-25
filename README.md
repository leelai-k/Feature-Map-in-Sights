# feature-map-in-Sights
Visualize feature maps activations of a trained Convolutional Network by projecting activations back onto an input image's pixels using a Deconvolutional network.

Core implementation based on:
Zeiler M., Fergus R. Visualizing and Understanding Convolutional Networks. In ECCV 2014.
https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53

VISUALIZATION MODES
IMAGE: python visualize_main.py images/tiger.jpeg -out defaults/car_vis_default -m 0.2 -a 9
	<insert-grid-img-here>
GIF: python visualize_main.py images/tiger.jpeg -g -out tiger_gifs/tiger
	<insert-gif-here>

EXAMPLES
Pre-generated examples and scripts.

inputs: \images
outputs:
	gifs: results\gifs
	images: results\images
scripts:
	generate_gifs_vis.sh
	generate_image_vis.sh

USAGE
--------------------------------------------------------------
img_path(req): Image to be visualized. Currently only JPEG is guaranteed support. Img will be resized to 224 x 224 and normalized.
--------------------------------------------------------------
output_path(opt): Relative path from featuremap-InSights/results where visualization(s) are stored. Don't include file extension in path, will default to JPEG.
--------------------------------------------------------------
gif(opt): If True, generates gif visualization,
If False generates image visualization.
--------------------------------------------------------------
	GIF
--------------------------------------------------------------
	Visualizes projections of feature map(s) from each layer to input space progressing from a single feature map to all feature maps in a layer. 

	All activations are kept and feature maps are added to gif from highest to lowest max(activation) per layer.
--------------------------------------------------------------
	IMAGE
--------------------------------------------------------------
	Determines which activations in which feature maps are visualized for an img.
--------------------------------------------------------------
	activations(opt): Total # of activations to keep per map.
--------------------------------------------------------------
	maps(opt): Total # of maps to keep per layer.
--------------------------------------------------------------
        * Nonzero integers represents specific count. 
        * Decimal btwn 0 and 1 represent %.
        * All other inputs default to a single instance. 
        * Sorted by max_activation.
--------------------------------------------------------------
Paper vs FeatureMap-InSights:
Paper
* Visualizes single feature maps for a set number of top activations per layer
* Uses Krizhevksy et al CNN

FeatureMap-InSights
* Focuses on Visualizes ranges of feature maps for ranges of activations together per layer.
* Uses pre-trained Vgg16 CNN (In process of expanding support to  general CNN architectures)
* Includes GIF visualization.

FAQ

Why visualize ranges of feature maps and activations instead of just the top feature maps and activations?

Since CNNs are nonlinear the superposition principle does not apply and just looking at individual feature maps separately is not enough to understand the total effects of a feature map on a network. A projection from multiple combined feature maps could produce behavior not found by just combining the projections of individual feature maps.

Why sort feature maps by max(activation) instead of another metric?

Max(activation) was chosen in the scenario primarily for it's simplicity. Other metrics could potentially be better suited to this problem. Future iterations to this project will experiment with other metrics.

Why is there greater noise in the projections from later layers?

The UnPooling block in the DCNN(Deconvolutional Network) is not a perfect reversal of the pooling block. Only the indices of the max value are saved, so the other values are lost adding noise. Projections from later layers experience this effect more as they pass through more UnPooling blocks.

Does the Deconvolution Network actually use deconvolution?

No mathematical deconvolution is not used, the network makes use of Transpose Convolution blocks to reverse convolution blocks. However both terms are used across this project since both terms appear in the literature. 

Future Work
* Expand supported CNNs.
* Allow for other metrics besides Max(activation) (e.g. Highest variance btwn activations)
* Add support for classifiers and occlusion analysis as outlined in Zeiler et al.
* Add support for more objective measures such as Mutual Information between projection and input image for different sets of feature maps.