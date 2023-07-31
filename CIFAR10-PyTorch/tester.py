import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

