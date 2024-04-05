import statistics

# List of F1 scores
f1_scores = [45.42, 41.02, 33.42, 38.69, 34.34, 50.46, 47.74, 25.08, 37.21, 44.76]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
