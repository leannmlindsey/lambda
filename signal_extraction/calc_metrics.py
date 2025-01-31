import pandas as pd

# Read the CSV data
data = pd.read_csv('results_table.csv')  # Replace with your actual filename

# Calculate averages for each metric column
# Exclude the 'Filename' and 'Algorithm' columns since they're not numeric
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC']
averages = data[metrics].mean()

# Print the results
print("Average Metrics:")
print("-" * 40)
for metric, value in averages.items():
    print(f"{metric}: {value:.6f}")
