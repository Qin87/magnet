import statistics

# List of F1 scores
f1_scores = [43.74, 47.34, 41.67, 47.46, 37.93, 32.34, 40.72, 33.33, 47.05, 50.95]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
