import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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

# device = torch.device('cpu')
path = "state_dict_model_mnist.pt"
loaded_model = ConvNet()
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