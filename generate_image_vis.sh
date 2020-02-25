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
  python visualize_main.py images/"${i}".jpeg -out defaults/"${i}"_vis_default -m 0 -a 0
  python visualize_main.py images/"${i}".jpeg -out all_maps/"${i}"_vis_100_mp -a 0
  python visualize_main.py images/"${i}".jpeg -out all_activations/"${i}"_vis_100_ap -m 0
  python visualize_main.py images/"${i}".jpeg -out all_maps_all_activations/"${i}"_vis_100_np_100_ap 
  
done