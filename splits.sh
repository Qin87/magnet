for log_file in $(ls -1 SHAGAT_dgl_cora_Aug*.log | sort); do
  echo "     "
  grep -E '^(AP|G|Ch|F|S)' "$log_file" 
 #grep -v '^[eEtNA]' "$log_file"
  #tail -n 2 "$log_file"
done

