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
  # Single Activation Single Map
  python visualize_main.py images/"${i}".jpeg -out defaults/"${i}"_vis_default --maps 0 --activations 0
  # Single Activation All Maps
  python visualize_main.py images/"${i}".jpeg -out all_maps/"${i}"_vis_100_mp --activations 0
  # All Activations Single Map
  python visualize_main.py images/"${i}".jpeg -out all_activations/"${i}"_vis_100_ap --maps 0
  # All Activations All Maps
  python visualize_main.py images/"${i}".jpeg -out all_maps_all_activations/"${i}"_vis_100_mp_100_ap 
  
done