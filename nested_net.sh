#!/bin/bash

# List of AugDirect values
#augdirect_values="2 4 20 21 22 23 231 2311"
#net_values="addSym Sym addSympara"
net_values="DiG DiGib DiGSymib DiGSymCatib DiGSymCatMixib DiGSymCatMixSymib  DiGub DiGi3 DiGi4 DiGu3 DiGu4
addSym Sym addSympara
Mag MagQin Sig Qua
GCN GAT APPNP GIN Cheb SAGE
JKNet GPRGNN pgnn mlp sgc jk"
layer_values="1 2 3 4 5 6 7"

Direct_dataset='dgl/cora'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
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
    nohup python3 All2Main.py --AugDirect=0 --net=$net \
    --layer=$layer  --q=0  --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
      >Aug0${net}_${Direct_dataset_filename}_layer${layer}_T${timestamp}.log &
    pid=$!

    wait $pid
  done
done
