import statistics

# List of F1 scores
f1_scores =[66.98, 57.89, 58.69, 68.75, 69.80, 68.72, 78.18, 70.59, 59.21, 71.69]
# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
