#!/bin/bash

net_values="Mag "
q_value=0
layer_values=" 2 3 4 5 "
# layer_values=" 0 0.5 1 "
imbal="100  "


# 'citeseer_npz/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'
Direct_dataset=(  'WikipediaNetwork/chameleon')
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset   --net="$net"    --layer="$layer"
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        # for imba_value  in $imbal; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
            python3 main.py   --use_best_hyperparams=1   --num_split=10   --seed="$layer" \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
          # done
        done
        done
done