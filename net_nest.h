#!/bin/bash

net_values="QiG QiGi2 QiGu2 QiGi3 QiGu3 QiGi4 QiGu4"
q_value=0
layer_values="1 2 3 4"

# Datasets to iterate over
Direct_dataset=( 'citeseer_npz/' 'cora_ml/' 'telegram/telegram' )

# Generate a timestamp for log file naming
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
            python3 main.py --net="$net" --W_degree=5 --layer="$layer" --q="$q_value" --Dataset="$Didataset" > "NoSelfLoop_${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

        done
    done
done