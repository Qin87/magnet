#!/bin/bash

net_values=" ScaleNet  "
q_value=0
layer_values=" 2 3 "
imbal="4  "
Dir="1"

# 'Cora/' 'CiteSeer/' 'PubMed/' 'dgl/photo' 'dgl/computer' 'dgl/reddit' 'dgl/coauthor-cs' 'dgl/coauthor-ph' 'dgl/Fyelp' 'dgl/Famazon'
Direct_dataset=( 'citeseer_npz/'  )  # 'cora_ml/'  'citeseer_npz/'   'WikipediaNetwork/squirrel, WikipediaNetwork/chameleon'
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset     # --alphaDir="$alphadir" --betaDir="$betadir" --gamaDir="$gamadir"
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for alphadir in $Dir; do
        for betadir in $Dir; do
        #for gamadir in $Dir; do
# //        for imba_value  in $imbal; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_Dir${betadir}__${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
python3 main.py   --jk='max' --BN_model=1   --lr=0.01 --dropout=0.0  --betaDir="$betadir" \
 --net="$net"  --layer="$layer"   --Dataset="$Didataset" > "$log_output" \
             2>&1
            wait $pid
          done
        #done
       #done
       done
       done
       done
done