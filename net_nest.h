#!/bin/bash

# List of AugDirect values
#net_values="DiGSymib DiGSymCatib Qua Sig DiG DiGib DiGSymib DiGSymCatib DiGSymCatMixib"
# DiGi4 DiGu3 DiGu4
#addSym Sym addSympara
#Mag MagQin Sig Qua
#GCN GAT APPNP GIN Cheb SAGE
#JKNet pgnn mlp sgc"Cheb MagQin DiGSymib DiGSymCatib  # --MakeImbalance
net_values=" MagQin"

layer_values="1 2  3 4  "
aug_values="0 "

Direct_dataset=" telegram/telegram "  # Update your Direct_dataset value
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
  for direct_data in $Direct_dataset; do
    for net in $net_values; do
      for aug in $aug_values; do
        nohup python3 All2MainStop.py --AugDirect=$aug --net=$net --MakeImbalance  --IsDirectedData  \
        --layer=$layer  --q=0  --Direct_dataset="$direct_data"  \
          > ${Direct_dataset_filename}ImbalaDirect_${timestamp}_Aug${aug}${net}_layer${layer}q0.log &
        pid=$!

        wait $pid
      done
    done
  done
done
