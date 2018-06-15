# DOC-tf
Implementation of Deep Occlusion Estimation in Tensorflow

## DOC: Deep OCclusion Estimation From a Single Image

This repository is based on the Caffe model found [here](https://github.com/pengwangucla/DOC). The goal was to convert this model to Tensorflow for further network dissection and experimentation.

## Model Conversion

The first step is to get the caffe model weights and prototxt definition file. They can be found [here](https://drive.google.com/file/d/0B7DaWBKShuMBN0drTzRRMlpoTmc/view). I first tried converting the model from Caffe to Tensorflow using the [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) tool. Unfortunately, many layers (including Crop and Deconvolution) are not properly implemented. I then came across the [MMdnn](https://github.com/Microsoft/MMdnn) tool from Microsoft, which allows for conversion of models between different frameworks (e.g. Tensorflow, Caffe, Pytorch, etc.) I adapted the code there to perform conversion of the Caffe model into a Tensorflow graph. Here are some of the issues I ran into:

1. I didn't want to install Caffe just to do the conversion, so I used a pure protobuf implementation. You have to uncomment out the corresponding lines in the mmdnn/conversion/caffe/resolver.py file to fallback on a protobuf implementation.

2. The converter expects the crop layer to have an output shape, which is missing from the .prototxt file. I just commented out the crop layers in the .prototxt file.

3. The converter expects the deconvolution layer to have a dilation parameter, which is missing from the .prototxt file. I just set it to 1 by default in the mmdnn/conversion/caffe/shape.py file.

4. There was also an issue with computing the max size of the output layers, since the output prediction is a zero-dim numpy scalar. I edited mmdnn/conversion/caffe/graph.py to check for zero-dim numpy shapes.

Following these steps resulted in a numpy (.npy) file containing the network weights, and a python (.py) file containing the model graph definition.

## Tensorflow Graph

The final output Tensorflow graph (\*.py file) had a few extra "L"s in the shape components, which I had to manually delete before the python file ran without errors. I also had to go in and change the shapes of some of the layers to allow for arbitrary input image sizes (it was set to 500x500 before). The main changes were making the input layer accept arbitrary dimension images and padding with 1 instead of 35. Also, in the deconvolution layers, I checked for the input image size dynamically and used "SAME" padding instead of "VALID" padding.

## Example Inference

You can pass an arbitrary image to the model and get the model's edge and/or orientation outputs.
