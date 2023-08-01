import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import onnx
import onnxruntime as ort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

#Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

#Transform dataset to tensors and normalize to range [1,1]

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

#CIFAR10: 60000 32x32 color images in 10 classes with 6000 images per class

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train = True,
    download = True,
    transform = transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train = False,
    download = True,
    transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = batch_size,
                                           shuffle=False)


class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,32,3)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32,64,3)
    self.conv3 = nn.Conv2d(64,64,3)
    self.fc1 = nn.Linear(64*4*4, 64)
    self.fc2 = nn.Linear(64,10)

  def forward(self,x):
    #N, 3, 32, 32
    x.shape
    x = self.conv1(x)
    x.shape
    x = self.relu(x)
    x = self.pool(x)
    x.shape
    x = self.conv2(x)
    x.shape
    x = self.relu(x)
    x = self.pool(x)
    x.shape
    x = self.conv3(x)
    x.shape
    x = self.relu(x)
    x = torch.flatten(x,1)
    x.shape
    x = self.fc1(x)
    x.shape
    x = self.relu(x)
    x = self.fc2(x)
    x.shape
    return x

model = ConvNet().to(device)

def training():

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr  = learning_rate)

  n_total_steps = len(train_loader)
  for epoch in range(num_epochs):

    running_loss = 0.0
    for i, (images,labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item() #Avg loss, we can output the absolute loss for each epoch

    print(f'[{epoch+1}] loss: {running_loss / n_total_steps:.3f}')

  print('Finished Training!')
  path = "state_dict_model_cifar10-bs64.pt"
  torch.save(model.state_dict(), path)

def pytorch_inference():
  path = "state_dict_model_cifar10-bs32.pt" # Try using learned weights for different batch sizes
  loaded_model = ConvNet().to(device)
  loaded_model.load_state_dict(torch.load(path))
  loaded_model.eval()

  with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)
    total_inference_time = 0

    for images, labels in test_loader:
      start_time = time.time()
      outputs = loaded_model(images)
      end_time = time.time()
      inference_time = end_time - start_time
      total_inference_time += inference_time

      _, predicted = torch.max(outputs,1) #Return the max value and index of all the elements in the input tensor i.e., highest probablity
      n_correct += (predicted == labels).sum().item()

    acc = 100.0 * (n_correct/n_samples)
    print(f'Accuracy of the model: {acc:.4f}%')

    average_inference_time = total_inference_time / len(test_loader)
    print(f"Average Inference time: {average_inference_time:.4f} seconds")

def onnx_converter():
  path = "state_dict_model_cifar10-bs32.pt"
  loaded_model = ConvNet().to(device)
  loaded_model.load_state_dict(torch.load(path))
  loaded_model.eval()
  dummy_input = torch.randn(batch_size, 3, 32, 32, requires_grad=True).to(device)

  torch.onnx.export(loaded_model, 
                    dummy_input,
                    'model.onnx',
                    export_params=True,        # store the trained parameter weights inside the model file
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],
                    output_names=['output'],
                    verbose=True,
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})

  # Load the ONNX model
  model = onnx.load("model.onnx")

  # Check that the model is well formed
  onnx.checker.check_model(model)

  # Print a human readable representation of the graph
  print(onnx.helper.printable_graph(model.graph))


def onnx_inference():
  n_correct = 0
  n_samples = len(test_loader.dataset)
  total_inference_time = 0

  ort_sess = ort.InferenceSession('model.onnx', None, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])

  # Fetch input and output tensor names
  input_name = ort_sess.get_inputs()[0].name
  output_name = ort_sess.get_outputs()[0].name

  for images, labels in test_loader:

      input_array = images.numpy().astype(np.float32)

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
  print(f"Accuracy: {acc:.4f}%")

  average_inference_time = total_inference_time / len(test_loader)
  print(f"Average Inference time: {average_inference_time:.4f} seconds")

if __name__ == '__main__':
  # training()
  pytorch_inference()
  onnx_converter()
  onnx_inference()