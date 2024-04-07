import statistics

# List of F1 scores
f1_scores = [65.74, 49.98, 55.26, 62.34, 46.48, 37.90, 44.33, 34.24, 54.16, 51.21]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
