# Image Segmentation Using Gaussian Mixture Model with PyTorch

## Environment

* cpu: i7-9750h
* gpu: GTX 1660 Ti
* python 3.7.4
* pytorch: 1.5.0

Implement a Gaussian mixture model (GMM) and apply it in image segmentation. First, use the K-means algorithm to find K central pixels. Second, use Expectation maximization (EM) algorithm to optimize the parameters of the model.
You can see the derivation of the mathematical formula on the following webpage(my medium account):
* EM algorithm: https://roger010620.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-em%E6%BC%94%E7%AE%97%E6%B3%95-expectation-maximization-algorithm-%E4%B8%80-ff03bd838c0a
* K means: https://roger010620.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-em%E6%BC%94%E7%AE%97%E6%B3%95-expectation-maximization-algorithm-%E4%BA%8C-k-means-3572aa4bd5d5
* Gaussian Mixture Model: https://roger010620.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-em%E6%BC%94%E7%AE%97%E6%B3%95-expectation-maximization-algorithm-%E4%B8%89-%E9%AB%98%E6%96%AF%E6%B7%B7%E5%92%8C%E6%A8%A1%E5%9E%8Bgaussian-mixture-model-gmm-84286c2d64c7
## Original Image
![Original Image](hw3_3.jpeg)
## K means output
K-means will first compute k (user defined) mean values of the input image and output the same image except the pixel values are scale to the nearest mean.
K = 5                         | K = 7 
:-----:                       |:-----:
![Original Image](result/k-means(pytorch)_5.png) | ![Original Image](result/k-means(pytorch)_7.png) 
K = 10                        | K = 15
![Original Image](result/k-means(pytorch)_10.png) | ![Original Image](result/k-means(pytorch)_15.png)
## GMM output
Using the means from k-means as initial value for GMM, after perform EM algorithm, output the image scale to the most probable mean value.
K = 5                         | K = 7 
:-----:                       |:-----:
![Original Image](result/GMM_5.png) | ![Original Image](result/GMM_7.png) 
K = 10                        | K = 15
![Original Image](result/GMM_10.png) | ![Original Image](result/GMM_15.png)
