for log_file in $(ls -1 DiG_Amazon-Computers_Aug*.log | sort); do
  echo "        "
  tail -n 2 "$log_file"
done

