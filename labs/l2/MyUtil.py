import numpy as np 
import matplotlib.pyplot as plt

def PCA(matrix: np.array, k: int) -> np.array:
	# Reduce the data to 2D using PCA 
	# Zero mean data 
	matrix = matrix - np.mean(matrix)

	# Calculate covariance 
	Sigma = np.cov(matrix, rowvar=False)
	
	# Calculate eigen values
	# 	find k eigenvectors wth the top kth highest eigen values
	eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[sorted_indices]
	eigenvectors = eigenvectors[:, sorted_indices]

	# reduce to k = D = 2 
	W = eigenvectors[:, :k]
	Z = matrix @ W 

	return Z

# Plot Matrix Data
def plot_matrix(matrix: np.array, title: str) -> None:
	plt.figure()
	plt.title(title)

	plt.scatter(np.array(matrix[:,0]), np.array(matrix[:, 1]))

	# Add labels and title
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.legend()

	# Show the plot
	plt.grid(True)  # Optional: Add grid for better visualization
	plt.savefig(f'{title}.png')
