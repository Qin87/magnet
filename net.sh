###!/bin/bash

# List of AugDirect values
#augdirect_values="2 4 20 21 22 23 231 2311"

#net_values="DiG DiGib DiGSymib DiGSymCatib DiGSymCatMixib DiGSymCatMixSymib  DiGub DiGi3 DiGi4 DiGu3 DiGu4 "   # tested
#net_values="GCN GAT APPNP GIN Cheb SAGE"    # tested
net_values="Mag MagQin Sig Qua"  #  UGCL can't work
#net_values="JKNet GPRGNN pgnn mlp sgc jk "   # tested
#net_values="DiGSymCatib DiGSymCatMixib DiGSymCatMixSymib "

layer_values="1 2 3"

Direct_dataset='WebKB/texas'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
#net='DiGSymCatib'

generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

logfile="outforlayer4.log"
exec > $logfile 2>&1

# Iterate over each AugDirect value
for net in $net_values; do
  nohup python3 All2Main.py --AugDirect=0 --net=$net \
  --layer=4  --q=0  --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
   2>&1 >Aug0${net}_${Direct_dataset_filename}_layer4_T${timestamp}.log &
  pid=$!

  wait $pid
done