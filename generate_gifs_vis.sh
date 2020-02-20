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
  mkdir "results/gifs/${i}_gifs"
  python visualize_main.py images/"${i}".jpeg -g -gdir "${i}_gifs"
done