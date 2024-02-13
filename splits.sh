for log_file in $(ls -1 *WikiCS__Aug*.log | sort); do
  echo "     "
  grep -v '^[eEtN]' "$log_file"
  #tail -n 2 "$log_file"
done

