#!/bin/bash

net_values="QiG  QiGi2  QiGu2  QiGi3 QiGu3  QiGi4 QiGu4"
q_value=0
layer_values="1 2 3  4  "    #


Direct_dataset=( 'citeseer_npz/' 'cora_ml/'  'telegram/telegram'  )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each net value   --MakeImbalance  --IsDirectedData --to_undirected  for Didataset in $Direct_dataset; do All2MainStop.py
for Didataset in "${Direct_dataset[@]}"; do
        for layer in $layer_values; do
          logfile="outforlayer${layer}.log"
          exec > $logfile 2>&1  # Redirect stdout and stderr to log file
          for net in $net_values; do
            nohup python3  main.py   --net=$net   --W_degree=5       \
            --layer=$layer  --q=$q_value  --Direct_dataset="$Didataset"  \
              >NoSelfLoop${Direct_dataset_filename}_${timestamp}_${net}_layer${layer}q${q_value}.log &
            pid=$!

            wait $pid
          done
        done
done