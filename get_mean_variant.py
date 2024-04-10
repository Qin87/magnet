import statistics

# List of F1 scores
f1_scores =[58.21, 53.16, 60.12, 80.52, 66.08, 45.58, 68.20, 49.26, 49.62, 69.54]
# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
