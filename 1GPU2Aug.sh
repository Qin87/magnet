#!/bin/bash

# List of AugDirect values
augdirect_values="-1 2 4 20 21 22 23 231 2311"
Direct_dataset='WebKB/texas'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=True
unDirect_data='CiteSeer'
net='GAT'

generate_timestamp() {
  date +"%d%H%Ms%S"
}

# Function to wait for PIDs
wait_for_pids() {
  for pid in "$@"; do
    wait $pid
  done
}

# Iterate over each AugDirect value
i=0
for augdirect in $augdirect_values; do
  i=$((i + 1))
  if [ "$IsDirData" = False ]; then
    filename="${net}_${unDirect_data}"
  else
    filename="${net}_${Direct_dataset_filename}"
  fi

  # Run the process
  nohup python DiGMain.py --GPUdevice=2 --AugDirect=$augdirect --net=$net \
    --Direct_dataset=$Direct_dataset --undirect_dataset=$unDirect_data \
    > "${filename}_Aug${augdirect}_T$(generate_timestamp)_${i}.log" &

  # Store the last PID for waiting
  last_pid="$!"

  # Wait for the last PID to finish before moving to the next iteration (every 2 iterations)
  wait_for_pids $last_pid
done

