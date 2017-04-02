# PCA on MNIST

[//]: # (Image References)
[image1]: ./output_images/correctsample.jpg "Correct Sample"
[image2]: ./output_images/wrongsample.jpg "Wrong Sample"

## Overview

pca is a Python 2.7 application, mainly to classify the handwriting characters from [MNIST data set](http://yann.lecun.com/exdb/mnist/). The data file contains 3,823 samples, which are 1 x 64 vectors. Each sample is grayscale 8x8 handwriting images and the label of the digit.

The first ten samples of each digit selected for training, and the remaining samples used for testing. The testing samples were classified using the nearest neighbor classification with euclidean distance with the transformed values in the eigenspace.


## The Project

Three classifications were performed using three different lambda values, 40%, 60%, and 80%. These values corresponded to the proportion of the sum the selected eigenvalues to the sum of all eigenvalues. The following table shows the accuracy values for the lambda values and the digits.

| Digit |  %40   |  %60   |  %80   |
|:-----:|:------:|:------:|:------:|
|   0   | 70.49% | 80.60% | 98.36% |
|   1   | 25.07% | 85.49% | 85.49% |
|   2   | 77.57% | 85.14% | 91.08% |
|   3   | 36.41% | 49.87% | 92.08% |
|   4   | 33.42% | 68.97% | 79.58% |
|   5   | 37.98% | 32.51% | 84.97% |
|   6   | 55.86% | 93.73% | 96.73% |
|   7   | 51.19% | 89.92% | 94.16% |
|   8   | 23.78% | 50.27% | 77.03% |
|   9   | 15.32% | 19.62% | 53.49% |

The following figure shows a correctly predicted sample.
![alt text][image1]


The following figure shows a wrongly predicted sample.
![alt text][image2]


## Dependencies
* numpy library is required to perform matrix operations.
* matplotlib.pyplot library is required to plot the images.
* matplotlib.backends.backend_pdf.PdfPagesscipy function is required to convert the plots to pdf.
