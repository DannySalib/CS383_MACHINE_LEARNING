# Danny Salib
# Jan 15 2025
# CS383 ML 
# Lab 2

import os 
import numpy as np 
from PIL import Image
import MyUtil
import matplotlib.pyplot as plt

PATH = './yalefaces'

def main():
	
	# STEP 1 - read each image, resize, flatten, and append to our data matrix
	X = [] # initialize data matrix 

	files = os.listdir(PATH)
	for f in files:

		if f.endswith('.txt'):
			continue # skip the read me file 

		img = Image.open(f'{PATH}/{f}')
		img = img.resize((40,40)) # convert to 40x40 resolution 
		Xi = np.asmatrix(img.getdata(),dtype=np.float64) # flatten 
		
		# We have our subset X_i, append it to our matrix 
		if len(X) == 0:
			X = Xi
		else:
			X = np.append(X, Xi, axis=0)

	# Plot initial matrix 
	MyUtil.plot_matrix(X, 'Initial Data')

		# Reduce the data to 2D using PCA 
	# Zero mean data 
	X = X - np.mean(X)

	# Calculate covariance 
	Sigma = np.cov(X, rowvar=False)
	
	# Calculate eigen values
	# 	find k eigenvectors wth the top kth highest eigen values
	eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[sorted_indices]
	eigenvectors = eigenvectors[:, sorted_indices]

	# reduce to k = D = 2 
	W = eigenvectors[:, :2]
	Z = X @ W 

	# Plot resulting matrix 
	MyUtil.plot_matrix(Z, 'PCA (2D) Data')

	# project subject02.centerlight onto the k most important principle components
	matrix = Image.open(f'{PATH}/subject02.centerlight')
	matrix = matrix.resize((40, 40))
	
	# Save initial image
	plt.figure()
	plt.imshow(matrix)
	plt.savefig('initial.png')
	
	matrix = np.asmatrix(matrix.getdata(),dtype=np.float64) # flatten 
	# Zero mean image 
	matrix_mean = np.mean(matrix)
	matrix = matrix - matrix_mean
	
#	Determines the smallest k such that the k largest eigenvalues constitute at least 95% of the
#	eigenvalues.
#	is_
	cumulative_sum = np.cumsum(eigenvalues)
	total_sum = np.sum(eigenvalues)
	threshold = 0.95 * total_sum
	# Dimensionality = k
	k = np.searchsorted(cumulative_sum, threshold) + 1  # +1 because indices are 0-based
	W = eigenvectors[:, :k]

	Z = np.dot(matrix, W)
	
	Xhat = Z @ W.T

	# Undo zero mean and convert to 40x40 image 
	Xhat += matrix_mean
	Xhat = Xhat.reshape((40, 40))

	# Save result
	plt.figure()
	plt.imshow(Xhat)
	plt.savefig('result.png')

	
if __name__ == '__main__':
	main()
