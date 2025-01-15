# Danny Salib
# Jan 15 2025
# CS383 ML 
# Lab 2

import os 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
import MyUtil

PATH = './yalefaces'

def main():
	
	# STEP 1 - read each image, resize, flatten, and append to our data matrix
	X = [] # initialize data matrix 

	files = os.listdir(PATH)
	for f in files[1:]: # ignore README

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

	Z = MyUtil.PCA(X, k=2)	# 2D representation of data
	
	# Plot initial matrix 
	MyUtil.plot_matrix(Z, 'PCA (2D) Data')
	
if __name__ == '__main__':
	main()
