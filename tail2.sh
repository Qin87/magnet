for log_file in $(ls -1 GAT_PubMed_Aug*.log | sort); do
  echo "        "
  tail -n 2 "$log_file"
done

