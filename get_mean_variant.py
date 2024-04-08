import statistics

# List of F1 scores
f1_scores = [55.56, 51.03, 39.26, 57.01, 47.41, 38.23, 48.06, 44.44, 52.04, 42.95]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
