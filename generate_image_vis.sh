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

for i in "${inputs[@]}"; do
  python visualize_main.py images/"${i}".jpeg -out images/defaults/"${i}"_vis_default -mp 0 -ap 0
  python visualize_main.py images/"${i}".jpeg -out images/all_maps/"${i}"_vis_100_mp -ap 0
  python visualize_main.py images/"${i}".jpeg -out images/all_activations/"${i}"_vis_100_ap -mp 0
  python visualize_main.py images/"${i}".jpeg -out images/all_maps_all_activations/"${i}"_vis_100_np_100_ap 
  
done