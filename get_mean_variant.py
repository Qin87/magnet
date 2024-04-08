import statistics

# List of F1 scores
f1_scores = [36.75, 28.70, 23.72, 31.02, 32.37, 32.24, 44.23, 25.32, 35.83]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
