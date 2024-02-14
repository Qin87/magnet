for log_file in $(ls -1  Norelulayer1Cheb_dgl_citeseer_Aug* | sort); do
  echo "        "
  tail -n 2 "$log_file"
done

