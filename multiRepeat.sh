#!/bin/bash

# List of AugDirect values
augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
GPUdevice_values="0 1 2"
#GPUdevice_values="0 1 2 3"
Direct_dataset='citeseer_npz/'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='Amazon-Computers'
net='GAT'

generate_timestamp() {
  date +"%M"
}

random_digit=$(shuf -i 0-9 -n 1)

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
                                                            [ Read 52 lines ]
^G Get Help            ^O WriteOut            ^R Read File           ^Y Prev Page           ^K Cut Text            ^C Cur Pos
^X Exit                ^J Justify             ^W Where Is            ^V Next Page           ^U UnCut Text          ^T To Spell

