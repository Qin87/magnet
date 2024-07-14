#!/bin/bash

net_values="  QiG GCN QiGi2"
q_value=0
layer_values=" 2 3    "    #


Direct_dataset=( 'cora_ml/' 'citeseer_npz/'  'dgl/cora' )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
unDirect_data='PubMed'
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each net value   --MakeImbalance  --IsDirectedData --to_undirected  for Didataset in $Direct_dataset; do All2MainStop.py
for Didataset in "${Direct_dataset[@]}"; do
        for layer in $layer_values; do
          logfile="outforlayer${layer}.log"  # Adjust log file name with layer number
            exec > $logfile 2>&1  # Redirect stdout and stderr to log file
          # Iterate over each layer value
          for net in $net_values; do
            nohup python3  main.py   --net=$net   --W_degree=5       \
            --layer=$layer  --q=$q_value  --Direct_dataset="$Didataset"  --undirect_dataset="$unDirect_data" \
              >NoSelfLoop${Direct_dataset_filename}_${timestamp}_${net}_layer${layer}q${q_value}.log &
            pid=$!

            wait $pid
          done
        done
done