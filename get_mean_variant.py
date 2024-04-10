import statistics

# List of F1 scores
f1_scores = [60.18, 63.74, 63.53, 60.87, 60.62, 54.37, 66.44, 59.89, 61.93, 71.94]



# Calculate the average (mean) of F1 scores
average = statistics.mean(f1_scores)

# Calculate the standard deviation of F1 scores
std_dev = statistics.stdev(f1_scores)

# Print the result in the specified format
print(f"{average:.3f}±{std_dev:.2f}")
