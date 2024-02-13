for log_file in $(ls -1  GIN_dgl_citeseer_Aug*T132158s08.log | sort); do
  echo "        "
  tail -n 2 "$log_file"
done

