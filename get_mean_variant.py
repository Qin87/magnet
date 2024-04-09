import statistics

# List of F1 scores
f1_scores = [62.67, 44.89, 43.03, 65.31, 48.78, 62.50, 47.27, 46.35, 57.25, 52.77]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
