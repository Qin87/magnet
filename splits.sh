for log_file in $(ls -1   *.log | sort); do
  echo "     "
  grep -E '^(AP|GI|Ch|F|S|Di|GC|M)' "$log_file" 
 #grep -v '^[eEtNA]' "$log_file"
  #tail -n 2 "$log_file"
done

