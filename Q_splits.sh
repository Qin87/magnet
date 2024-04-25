for log_file in $(ls -1   *.log | sort); do
  echo "     "
  echo "$log_file"
 grep -E '^(AP|GI|F|S|Di|GC|M|a|A)' "$log_file"
 #grep -v '^[eEtNA]' "$log_file"
  tail -n 1 "$log_file"
done

