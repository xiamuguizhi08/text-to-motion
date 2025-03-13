#!/bin/bash

# Change to the specified directory
cd /liujinxin/code/text-to-motion/

# Accept arrays from command line arguments (comma-separated values)
IFS=',' read -r -a input_folders <<< "$1"
IFS=',' read -r -a output_folders <<< "$2"
IFS=',' read -r -a pred_roots <<< "$3"

# Check if the arrays are the same length
if [ ${#input_folders[@]} -ne ${#output_folders[@]} ] || [ ${#input_folders[@]} -ne ${#pred_roots[@]} ]; then
  echo "Error: Arrays have different lengths!"
  exit 1
fi

# Verify that the input folders, output folders, and pred roots exist
for i in "${!input_folders[@]}"; do
  echo "Checking input folder: ${input_folders[$i]}"
  echo "Checking output folder: ${output_folders[$i]}"
  echo "Checking pred root: ${pred_roots[$i]}"

done

# Iterate over the arrays and run demo.py with different parameters in parallel
for i in "${!input_folders[@]}"; do
  echo "Running demo.py for input: ${input_folders[$i]}, output: ${output_folders[$i]}, pred_root: ${pred_roots[$i]}"
  /liujinxin/anaconda3/envs/RDT/bin/python eval_n.py \
    --input_folder "${input_folders[$i]}" \
    --output_folder "${output_folders[$i]}" \
    --pred_root "${pred_roots[$i]}" &
done

# Wait for all background processes to finish
wait
echo "All processes have completed."
