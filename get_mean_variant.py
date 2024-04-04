import statistics

# List of F1 scores
f1_scores = [62.10, 55.95, 38.69, 57.03, 55.14, 48.51, 62.25, 42.67, 47.83, 42.17]


# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
