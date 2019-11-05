# CS6375ML-PCA
Use Principal Component Analysis to compress the same images

The package used and installation instructions
==============================================
We use some image processing and machine learning libraries to implement color quantization by using PCA algorithm.

Their names and installation instructions are as follows:
1. numpy
	Open the terminal and input: pip3 install numpy
2. matplotlib
	Open the terminal and input: pip3 install matplotlib
3. imageio
	Open the terminal and input: pip3 install imageio
4. PIL
	Open the terminal and input: pip3 install pillow

Development environment
=======================
Python 3.7.2

Algorithm and compile
=====================
We use PCA algorithm to quantize the images. Firstly using numpy to transform the original image to 2D array on RGB and then reconstruct 2D matrix using PCA algorithm. Finally, combine RGB to color image and using matplotlib to display the quantized images with different values.
