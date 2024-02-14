for log_file in $(ls -1   Norelulayer3Cheb_dgl_citeseer_Aug*_T141435s51.log | sort); do
  echo "        "
  grep '^C' "$log_file"
  tail -n 1 "$log_file"
done

