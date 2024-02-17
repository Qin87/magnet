for log_file in $(ls -1 layer1SymDiGCN_dgl_citeseer_Aug*.log | sort); do
  echo "     "
  grep -E '^(AP|G|Ch|F|S)' "$log_file" 
 #grep -v '^[eEtNA]' "$log_file"
  #tail -n 2 "$log_file"
done

