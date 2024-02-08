for log_file in $(ls -1 DiG_WebKB_Cornell_Aug*.log | sort); do
  echo "     "
  grep -v '^[eEtN]' "$log_file"
  #tail -n 2 "$log_file"
done

