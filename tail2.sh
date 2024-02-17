for log_file in $(ls -1   layer2APPNP_dgl_citeseer_Aug* | sort); do
  echo "        "
  grep '^AP' "$log_file"
  tail -n 1 "$log_file"
done

