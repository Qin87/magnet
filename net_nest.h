#!/bin/bash

# CiG  CiGi2 CiGi3  CiGu2 CiGu3 CiGi4 CiGu4
# UiGi2 UiGi3 UiGi4 UiGu2 UiGu3 UiGu4 UiG
# IiGi2 IiGi3 IiGi4 IiGu2 IiGu3 IiGu4 IiG
# iiGi2 iiGi3 iiGi4 iiGu2 iiGu3 iiGu4 iiG
# TiGi2 TiGi3 TiGi4 TiGu2 TiGu3 TiGu4 TiG
# iiAi2 iiAi3 iiAi4 iiAu2 iiAu3 iiAu4 iiA   TiAi2 TiAi3 TiAi4 TiAu2 TiAu3 TiAu4 TiA  iiCi2 iiCi3 iiCi4 iiCu2 iiCu3 iiCu4 iiC   TiCi2 TiCi3 TiCi4 TiCu2 TiCu3 TiCu4 TiC  iiSi2 iiSi3 iiSi4
# iiSu2 iiSu3 iiSu4 iiS   TiSi2 TiSi3 TiSi4 TiSu2 TiSu3 TiSu4 TiS
net_values=" AiGs2 AiGs3  "
q_value=0
layer_values="4 5 6"
imbal="2  "

# 'Cora/' 'CiteSeer/' 'PubMed/' 'dgl/photo' 'dgl/computer' 'dgl/reddit' 'dgl/coauthor-cs' 'dgl/coauthor-ph' 'dgl/Fyelp' 'dgl/Famazon'
# 'citeseer_npz/' 'cora_ml/'  'telegram/telegram'  'citeseer_npz/' 'cora_ml/'  'telegram/telegram' 'dgl/pubmed' 'dgl/cora' 'WikiCS/'
Direct_dataset=( 'WikipediaNetwork/chameleon'  )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
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
        for imba_value  in $imbal; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
            python3 main.py   --feat_dim=128  --dropout=0.0  --lr=0.005 --l2=0.0 --net="$net"  --layer="$layer"   --epoch=15000  --NotImproved=400 --Dataset="$Didataset" > "$log_output" 2>&1
            wait $pid
          done
        done
        done
done