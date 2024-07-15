#!/bin/bash

net_values="Qym addQym addQympara"
q_value=0
layer_values="1 2 3  4  "    #

# 'citeseer_npz/' 'cora_ml/'  'telegram/telegram'
Direct_dataset=( 'citeseer_npz/' 'cora_ml/'  'telegram/telegram' 'dgl/pubmed' 'dgl/cora' 'WikiCS/'  )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            # Run the Python script with parameters and log output
            python3 main.py --net="$net"  --W_degree=5  --layer="$layer" --q="$q_value" --Dataset="$Didataset" > "NormNoQym_SelfLoop_${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            wait $pid
          done
        done
done