import numpy as np 
import matplotlib.pyplot as plt

# Plot Matrix Data
def plot_matrix(matrix: np.array, title: str) -> None:
	plt.figure()
	plt.title(title)

	plt.scatter(np.array(matrix[:,0]), np.array(matrix[:, 1]))

	# Add labels and title
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')

	# Show the plot
	plt.grid(True)  # Optional: Add grid for better visualization
	plt.savefig(f'{title}.png')
