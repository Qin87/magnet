#!/bin/bash

# List of AugDirect values
augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
GPUdevice_values="0 1 2"
#GPUdevice_values="0 1 2 3"
Direct_dataset='citeseer_npz/'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='Amazon-Computers'
net='GCN'

generate_timestamp() {
  date +"%M"
}

random_digit=$(shuf -i 0-9 -n 1)

# Lock file
lockfile="gpu_lock"

# Iterate over each AugDirect value
for augdirect in $augdirect_values; do
  for GPUdevice in $GPUdevice_values; do
    # Check if lock file exists for the GPU
    if [ ! -f "$lockfile$GPUdevice" ]; then
      # Create lock file for the GPU
      touch "$lockfile$GPUdevice"

      if [ "$IsDirData" = False ]; then
      filename="${net}_${unDirect_data}_Aug${augdirect}"
      else
      filename="${net}_${Direct_dataset_filename}_Aug${augdirect}"
      fi

      # Run the process
      nohup python main.py --GPUdevice=$GPUdevice --AugDirect=$augdirect  --net=$net \
	#--IsDirectedData	# if false interpret this line, True then keep
        --Direct_dataset=$Direct_dataset --undirect_dataset=$unDirect_data \
        > ${filename}_${timestamp}_$random_digit.log &

      # Release lock
      rm "$lockfile$GPUdevice"
    fi
  done
done

