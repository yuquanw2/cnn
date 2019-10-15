# DnCnn algorithm (Yujia Qiu(yujiaq3), Xiaodan Du(xdu12))

## Environment
tensorflow, python

## Train

	python dncnn.py 

Remember to modify the path of training data and test data in the dncnn.py.

## Test

	python test_cnn.py 

Remember to modify the path of test data in test_cnn.py

We are using image denoising using cnn and vae in the model. 15 conv layers + 1 deconv layers is our archetecture. This is inspired by the DnCnn model online. But the DnCnn model can only generate the picture with the same size as input. So we add a deconv layer to do the super resolution. About hyper parameter, I try a few parameters and decide to use 15 layers with 64 depth. We did not do the data preprocessing, we load the data and directly do the cnn.