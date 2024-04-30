#!/bin/bash

# List of AugDirect values
#net_values="DiGSymib DiGSymCatib Qua Sig DiG DiGib DiGSymib DiGSymCatib DiGSymCatMixib"
# DiGi4 DiGu3 DiGu4
#addSym Sym addSympara
#Mag MagQin Sig Qua
#GCN GAT APPNP GIN Cheb SAGE
#JKNet pgnn mlp sgc"Cheb MagQin DiGSymib DiGSymCatib
net_values="pgnn"

layer_values=" 2"

Direct_dataset='WebKB/texas'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')

generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each net value
for layer in $layer_values; do
  logfile="outforlayer${layer}.log"  # Adjust log file name with layer number
    exec > $logfile 2>&1  # Redirect stdout and stderr to log file
  # Iterate over each layer value
  for net in $net_values; do
    nohup python3 All2MainStop.py --AugDirect=0 --net=$net --MakeImbalance\
    --layer=$layer  --q=0  --Direct_dataset="$Direct_dataset"  \
      > ${Direct_dataset_filename}imBala_T${timestamp}_Aug0${net}_layer${layer}.log &
    pid=$!

    wait $pid
  done

done