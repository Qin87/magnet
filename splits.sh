for log_file in $(ls -1 *.log | sort); do
  echo "     "
  grep -E '^(AP|G|C|F)' "$log_file" 
 #grep -v '^[eEtNA]' "$log_file"
  #tail -n 2 "$log_file"
done

