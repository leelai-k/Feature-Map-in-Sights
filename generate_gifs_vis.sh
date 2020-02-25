#!/bin/bash
inputs=(
	car
	grasshopper
	henryviii
	mushroom
	paris
	pattern1
	stoplight
	tiger
	wolf)

# For all input images, generate and save deconv visualizations gifs.
# Feature Map Controls: 
# Activation fixed as %100 
# Feature maps varied from one to all feature maps. 
for i in "${inputs[@]}"; do
  python visualize_main.py images/"${i}".jpeg -gif -out "${i}"
done