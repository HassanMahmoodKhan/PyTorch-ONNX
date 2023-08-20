import onnx
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

transform = transforms.Compose(
    [transforms.ToTensor()])

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train = False,
    transform = transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = batch_size,
                                           shuffle=False)

n_correct = 0
n_samples = len(test_loader.dataset)
total_inference_time = 0

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

ort_sess = ort.InferenceSession('model.onnx', None, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Fetch input and output tensor names
input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

for images, labels in test_loader:

    input_array = images.numpy()
    start_time = time.time()
    outputs = ort_sess.run([output_name], {input_name: input_array})
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time

    # Process each image prediction separately (variable batch size)
    for i in range(input_array.shape[0]):
        predicted = np.argmax(outputs[0][i])  # Take the first output (if multiple outputs)
        n_correct += (predicted == labels[i].numpy()).sum().item()

acc = 100.0 * (n_correct/n_samples)
print(f"Accuracy: {acc:.2f}%")

average_inference_time = total_inference_time / len(test_loader)
print(f"Average Inference time: {average_inference_time:.4f} seconds")
