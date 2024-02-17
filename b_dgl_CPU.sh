###!/bin/bash

# List of AugDirect values
#augdirect_values="2 4 20 21 22 23 231 2311"
augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
#augdirect_values="0 1 -1 2"

<<<<<<< HEAD
Direct_dataset='dgl/cora'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
net='GCN'
=======
Direct_dataset='dgl/citeseer'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=True
unDirect_data='Cora'
net='SymDiGCN'
>>>>>>> a29cb825f1afa2c0afcc97e0ad0f85c851b51446

generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

logfile="outfor2.log"
exec > $logfile 2>&1

# Iterate over each AugDirect value
for augdirect in $augdirect_values; do
<<<<<<< HEAD
  nohup python3 DiGMain.py --AugDirect=$augdirect --net=$net \
<<<<<<< HEAD
    -to_undirected  \
    --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
    >SHA${net}_${Direct_dataset_filename}_Aug${augdirect}_T${timestamp}_ToUndi.log &
=======
=======
  nohup python3 Main.py --AugDirect=$augdirect --net=$net \
>>>>>>> 64109c357c22388df00a64b88166b0cbff5fd781
  --layer=1  --IsDirectedData    --Direct_dataset="$Direct_dataset" --undirect_dataset=$unDirect_data \
    >layer1${net}_${Direct_dataset_filename}_Aug${augdirect}_T${timestamp}.log &
>>>>>>> a29cb825f1afa2c0afcc97e0ad0f85c851b51446
  pid=$!

  wait $pid
done

# Optionally, wait for all background processes to finish
#wait

