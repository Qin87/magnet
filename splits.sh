for log_file in $(ls -1 GAT_PubMed_Aug*.log | sort); do
  echo "     "
  grep -v '^[eEtN]' "$log_file"
  #tail -n 2 "$log_file"
done

