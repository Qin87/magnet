import statistics

# List of F1 scores
f1_scores =[66.96, 68.24, 54.55, 52.44, 62.33, 79.33, 58.37, 57.68, 43.05, 57.95]
# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)
# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
