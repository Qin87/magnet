#!/bin/bash

# List of AugDirect values
augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
GPUdevice_values="2"
#GPUdevice_values="0 1 2 3"
Direct_dataset='WikiCS/'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=True
unDirect_data='Cora'
net='GAT'

generate_timestamp() {
  date +"%d%H%M"
}
timestamp=$(generate_timestamp)

logfile=$random_digit_"outfor2.log"
exec > $logfile 2>&1

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
        filename="${net}_${unDirect_data}"
      else
        filename="${net}_${Direct_dataset_filename}"
      fi

      # Run the process
      nohup python main.py --GPUdevice=$GPUdevice --AugDirect=$augdirect  --net=$net \
        --Direct_dataset=$Direct_dataset --undirect_dataset=$unDirect_data \
        > ${filename}_Aug${augdirect}_T${timestamp}.log &

      # Release lock
      rm "$lockfile$GPUdevice"

    fi
  done
done

