###!/bin/bash

# List of AugDirect values
#augdirect_values="2 4 20 21 22 23 231 2311"
#layer_values="0 1 -1 2 4 20 21 22 23 231 2311"
#set layer_values "1 2 3 4 5 6 7 8 9"
layer_values="1 2 3 4 5 6 7 8 9"
#augdirect_values="4 "

Direct_dataset='dgl/cora'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
net='DiGSymCatib'

generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

logfile="outfor2.log"
exec > $logfile 2>&1

# Iterate over each AugDirect value
for layer in $layer_values; do
  nohup python3 All2Main.py --AugDirect=0 --net=$net \
  --layer=$layer  --q=0  --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
    >Aug0${net}_${Direct_dataset_filename}_layer${layer}_T${timestamp}.log &
  pid=$!

  wait $pid
done