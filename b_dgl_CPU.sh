#!/bin/bash

# List of AugDirect values
#augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
augdirect_values="2311"

Direct_dataset='dgl/pubmed'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
net='GAT'

generate_timestamp() {
  date +"%d%H%M%S"
}
timestamp=$(generate_timestamp)

logfile="outfor2.log"
exec > $logfile 2>&1

# Iterate over each AugDirect value
for augdirect in $augdirect_values; do
 
  nohup python3 main.py --AugDirect=$augdirect --net=$net \
    --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
    > ${net}_${Direct_dataset_filename}_Aug${augdirect}_T${timestamp}.log &
done

# Optionally, wait for all background processes to finish
wait

