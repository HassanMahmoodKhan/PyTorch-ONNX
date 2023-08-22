## PyTorch - ONNX Conversion

A deep learning model that performs multi-class classification using the benchmark MNIST image data set.

### Script Instructions

There are two python script files i.e., mnist.py & onnx_runtime.py.

mnist.py performs the following steps:
- Importing relevant libraries
- Downloading the data set
- Performing data transformation
- Defining the convolutional neural network class 
- Model training
- PyTorch inference
- Model conversion to ONNX

onnx_runtime.py performs the following step:
- ONNX inference using ONNX Runtime

#### Hardware Specifics
- Processor: Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz   1.19 GHz
- Installed RAM: 12.0 GB
- System Type: 64-bit operating system, x64-based processor
- Operating System: Windows 11

#### Performance Comparison
The script is executed for a single test case. 

The table below describes the performance metrics recorded.

![image](https://github.com/HassanMahmoodKhan/PyTorch-ONNX/assets/97694796/86863dc7-9e88-461e-a885-12722943bc8a)

In terms of accuracy achieved (not important in the scope of this experiment), we can observe that both frameworks produce identical scores. Thus, there is no degradation.
However, with respect to inference time, there is a stark difference between the two. The graph below visualizes these performance gains when model inferencing with ONNX Runtime as opposed to PyTorch.

![image](https://github.com/HassanMahmoodKhan/PyTorch-ONNX/assets/97694796/ded8c516-68f9-449e-902a-c6425ff937fc)

Note: MNIST data set is readily available and part of all major frameworks such as PyTorch, TensorFlow. I have not added them in the github repository due to size limitation. You can simply download them using the script in the python file.



