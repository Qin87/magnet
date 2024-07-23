#!/bin/bash

# CiG  CiGi2 CiGi3  CiGu2 CiGu3 CiGi4 CiGu4
# UiGi2 UiGi3 UiGi4 UiGu2 UiGu3 UiGu4 UiG
# IiGi2 IiGi3 IiGi4 IiGu2 IiGu3 IiGu4 IiG
# iiGi2 iiGi3 iiGi4 iiGu2 iiGu3 iiGu4 iiG
# TiGi2 TiGi3 TiGi4 TiGu2 TiGu3 TiGu4 TiG
net_values=" QiGu3 TiGu3 UiGu3 "
q_value=0
layer_values="3"    #

# 'Cora/' 'CiteSeer/' 'PubMed/' 'dgl/photo' 'dgl/computer' 'dgl/reddit' 'dgl/coauthor-cs' 'dgl/coauthor-ph' 'dgl/Fyelp' 'dgl/Famazon'
# 'citeseer_npz/' 'cora_ml/'  'telegram/telegram'  'citeseer_npz/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'dgl/cora' 'WikiCS/'
Direct_dataset=( 'dgl/pubmed' 'dgl/cora')  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
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
            log_output="RemoveGenSelfLoop_QymNorm_NoSelfLoop_${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
            python3 main.py   --net="$net" --W_degree=5 --layer="$layer" --q="$q_value" --Dataset="$Didataset" > "$log_output" 2>&1
            wait $pid
          done
        done
done