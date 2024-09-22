import matplotlib.pyplot as plt

# Data from the table
x = [1000, 5000, 10000, 15000, 20000, 25000, 29000]
knn_accuracy = [0.6372, 0.7078, 0.7107, 0.7128, 0.7143, 0.7146, 0.7152]
mf_accuracy = [0.6443, 0.7331, 0.7362, 0.7378, 0.7390, 0.7394, 0.7409]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot KNN line
plt.plot(x, knn_accuracy, marker='o', linestyle='-', color='blue', label='KNN')

# Plot Matrix Factorization line
plt.plot(x, mf_accuracy, marker='s', linestyle='-', color='green', label='Matrix Factorization')

# Add labels and title
plt.title('Correctness Prediction Performance KNN vs. Matrix Factorization', fontsize=14)
plt.xlabel('Number of Training Questions', fontsize=12)
plt.ylabel('Prediction Accuracy', fontsize=12)

# Set the x-axis and y-axis limits
plt.xlim(0, 30000)
plt.ylim(0.63, 0.75)

# Add a grid for readability
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("correctness_prediction.png")