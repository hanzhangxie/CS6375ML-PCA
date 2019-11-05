import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# transform the image to 2D numpy array on RGB
def trans2D(imgName):
	# load the image
	img = imageio.imread(imgName)
	img = np.array(img)
	imgR = img[:, :, 0]
	imgG = img[:, :, 1]
	imgB = img[:, :, 2]

	return imgR, imgG, imgB

# reconstruct 2D matrix using PCA
def compress(img2D, p):
	# average by column
	average = np.mean(img2D, axis=1, keepdims=True)
	# minus the average
	meanRemove = img2D - average
	# covariance matrix
	covMat = np.cov(meanRemove)
	eigVal, eigVec = np.linalg.eigh(covMat)
	n = np.size(eigVec, axis=1)
	index = np.argsort(eigVal)
	index = index[::-1]
	eigVal = eigVal[index]
	eigVec = eigVec[:, index]
	if p*100 < n or p > 0:
		eigVec = eigVec[:, range(p*100)]
	selectVec = np.dot(eigVec.T, meanRemove)
	compressData = np.dot(eigVec, selectVec) + np.mean(img2D, axis=1, keepdims=True)
	compressData = np.uint8(np.absolute(compressData))

	return compressData 


if __name__ == '__main__':
	# display the compressed images
	transImage1 = trans2D("image1.jpg")
	# reconstruct RGB
	compressImg1R = compress(transImage1[0], 2)
	compressImg1G = compress(transImage1[1], 2)
	compressImg1B = compress(transImage1[2], 2)
	# combine RGB to color image
	colorImg1 = np.dstack((compressImg1R, compressImg1G, compressImg1B)) 
	compressImg1 = Image.fromarray(colorImg1)
	plt.figure(1)
	plt.axis('off')
	plt.imshow(compressImg1)
	plt.savefig('Figure_1.jpg')

	transImage2 = trans2D("image3.jpg")
	# reconstruct RGB
	compressImg2R = compress(transImage2[0], 8)
	compressImg2G = compress(transImage2[1], 8)
	compressImg2B = compress(transImage2[2], 8)
	# combine RGB to color image
	colorImg2 = np.dstack((compressImg2R, compressImg2G, compressImg2B)) 
	compressImg2 = Image.fromarray(colorImg2)
	plt.figure(2)
	plt.axis('off')
	plt.imshow(compressImg2)
	plt.savefig('Figure_2.jpg')

	transImage3 = trans2D("image5.jpg")
	# reconstruct RGB
	compressImg3R = compress(transImage3[0], 10)
	compressImg3G = compress(transImage3[1], 10)
	compressImg3B = compress(transImage3[2], 10)
	# combine RGB to color image
	colorImg3 = np.dstack((compressImg3R, compressImg3G, compressImg3B)) 
	compressImg3 = Image.fromarray(colorImg3)
	plt.figure(3)
	plt.axis('off')
	plt.imshow(compressImg3)
	plt.savefig('Figure_3.jpg')