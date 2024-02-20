for log_file in $(ls -1   layer1SAGE_dgl_citeseer_Aug* | sort); do
  echo "     "
  grep -E '^(AP|GI|Ch|F|S|Di)' "$log_file" 
 #grep -v '^[eEtNA]' "$log_file"
  #tail -n 2 "$log_file"
done

