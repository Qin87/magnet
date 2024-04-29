#!/bin/bash

# List of AugDirect values
#net_values="DiGSymib DiGSymCatib Qua Sig DiG DiGib DiGSymib DiGSymCatib DiGSymCatMixib"
# DiGi4 DiGu3 DiGu4
#addSym Sym addSympara
#Mag MagQin Sig Qua
#GCN GAT APPNP GIN Cheb SAGE
#JKNet pgnn mlp sgc"
net_values="DiGSymCatib DiGSymib addSympara DiGSymCatMixSymib DiGu3 DiGu4 DiGi4"

Direct_dataset='cora_ml/'  # Update your Direct_dataset value
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
    nohup python3 All2MainStop.py --AugDirect=20 --net=$net  --to_undirected \
    --layer=$layer  --q=0  --Direct_dataset="$Direct_dataset"  \
      > ${Direct_dataset_filename}True_T${timestamp}_Aug20${net}_layer${layer}.log &
    pid=$!

    wait $pid
  done
done
