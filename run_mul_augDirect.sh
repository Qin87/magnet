#!/bin/bash

# List of AugDirect values
augdirect_values="0 1 2 4 20 21 22 23 231 2311"
GPUdevice_values="0 1 2 3"
Direct_dataset='cora_ml/'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='Amazon-Photo'
net='SAGE'
lock_dir="/tmp/gpu_locks"

# Create lock directory if it doesn't exist
mkdir -p "$lock_dir"

# Function to acquire lock for a GPU
acquire_lock() {
    gpu_lock_file="$lock_dir/gpu$1.lock"
    exec 9>"$gpu_lock_file"
    flock -n 9 || exit 1
}

# Iterate over each AugDirect value
for augdirect in $augdirect_values; do
  for GPUdevice in $GPUdevice_values; do
    acquire_lock $GPUdevice || continue

    if [ "$IsDirData" = False ]; then
      filename="${net}_${unDirect_data}_Aug${augdirect}_GPU${GPUdevice}.log"
    else
      filename="${net}_${Direct_dataset_filename}_Aug${augdirect}_GPU${GPUdevice}.log"
    fi

    nohup python main.py --GPUdevice=$GPUdevice --AugDirect=$augdirect --IsDirectedData=$IsDirData \
      --Direct_dataset=$Direct_dataset --net=$net --undirect_dataset=$unDirect_data \
      > $filename &

    # Release the lock
    exec 9>&-
  done
done

