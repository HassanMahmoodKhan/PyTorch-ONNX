import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import time

batch_size = 1

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train = False,
    transform = transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = batch_size,
                                           shuffle=False)

n_correct = 0
n_samples = len(test_loader.dataset)
total_inference_time = 0

ort_sess = ort.InferenceSession('model.onnx', None, providers = ["CPUExecutionProvider"])

# Fetch input and output tensor names
input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

for images, labels in test_loader:

    input_array = images.numpy().astype(np.float32)

    start_time = time.time()
    # outputs = ort_sess.run(None, {'input': input_array})
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
print(f"Inference Time: {total_inference_time:.6f}")
