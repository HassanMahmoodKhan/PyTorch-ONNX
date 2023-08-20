## PyTorch - ONNX Conversion

A deep learning model that performs multi-class classification using the benchmark CIFAR10 image data set.

### Script Instructions

There is one python file i.e., cifar10.py, which contains all functions required for implementing the proposed methodology. These are:
- Importing relevant libraries
- Downloading the data set
- Performing data transformation
- Defining the convolutional neural network class 
- Model training
- PyTorch inference
- Model conversion to ONNX
- ONNX inference using ONNX Runtime

#### Hardare Specifics
This simulation was executed on a device with the following specifics:
- Processor: Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz   1.19 GHz
- Installed RAM: 12.0 GB
- System Type: 64-bit operating system, x64-based processor
- Operating System: Windows 11

#### Performance Comparison
I have executed the script for three test cases with respect to one of the model hyperparameters i.e., batch size. 

The tables below describe the performance metrics recorded.

![image](https://github.com/HassanMahmoodKhan/PyTorch-ONNX/assets/97694796/33321181-8b78-45b4-8ae1-d53ede4824b4)

Table 1 depicts metrics recorded when using the PyTorch framework for inference.

![image](https://github.com/HassanMahmoodKhan/PyTorch-ONNX/assets/97694796/0d27cefb-8364-4e3d-a98c-9832029ea434)

Table 2 depicts metrics recorded when using ONNX framework for inference.

As seen above, in terms of accuracy achieved (not important in the scope of this experiment), we can observe that both frameworks produce identical scores. 
Thus, there is no degradation.
However, with respect to inference time, there is a stark difference between the two for each test case. The graph below visualizes these performance gains when 
using ONNX with ONNX Runtime.

![image](https://github.com/HassanMahmoodKhan/PyTorch-ONNX/assets/97694796/6fd286f7-faa2-4dd6-9624-e8e94f010696)

Note: CIFAR10 data set is readily available and part of all major frameworks such as PyTorch, TensorFlow. I have not added them in the github repository
due to size limitation. You can simply download them using the script in the python file.
