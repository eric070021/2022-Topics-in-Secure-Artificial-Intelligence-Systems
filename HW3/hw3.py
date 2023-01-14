import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import pickle

# CKKS
# context = ts.context(
#             ts.SCHEME_TYPE.CKKS,
#             poly_modulus_degree=8192,
#             coeff_mod_bit_sizes=[40, 20, 20, 20, 40]
#           )
# context.global_scale = 2**20

context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
          )
context.global_scale = 2**40

# Plain moudule
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 32)
        self.output_fc = nn.Linear(32, output_dim)
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.input_fc(x)
        x = x * x # activation function
        y_pred = self.output_fc(x)
        return y_pred

# Encrypted model
# class Encrypted_MLP():
#     def __init__(self, input_dim, output_dim, weight):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = weight
#         self.input_weight = ts.ckks_tensor(context, self.weight['input_fc.weight'])
#         self.input_bias = ts.ckks_tensor(context, self.weight['input_fc.bias'])
#         self.output_weight = ts.ckks_tensor(context, self.weight['output_fc.weight'])
#         self.output_bias = ts.ckks_tensor(context, self.weight['output_fc.bias'])
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#         hidden = self.input_weight.mm(x.t()).transpose()
#         for i in range(batch_size):
#           hidden[i].add_(self.input_bias)
#         hidden.square_() # activation function
#         output = self.output_weight.mm(hidden.transpose()).transpose()
#         for i in range(batch_size):
#           output[i].add_(self.output_bias)
#         return output

class Encrypted_MLP():
    def __init__(self, input_dim, output_dim, weight):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = weight
        self.input_weight = ts.ckks_tensor(context, self.weight['input_fc.weight'])
        self.input_bias = ts.ckks_tensor(context, self.weight['input_fc.bias'])
        self.output_weight = ts.ckks_tensor(context, self.weight['output_fc.weight'])
        self.output_bias = ts.ckks_tensor(context, self.weight['output_fc.bias'])
    def forward(self, x):
        hidden = self.input_weight.mm(x.reshape(-1,1)).transpose() + self.input_bias
        hidden = hidden * hidden # activation function
        output = hidden.mm(self.output_weight.transpose()) + self.output_bias
        return output

# Download & Load EMNIST data
DOWNLOAD_MNIST = False

test_data = datasets.MNIST(root = './data', train = False, download = DOWNLOAD_MNIST, 
    transform = transforms.Compose([transforms.Resize(16), transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle = False, num_workers = 0)

# Load weights
weight = torch.load('model.pt')
model_plain = MLP(input_dim = 256, output_dim = 10)
model_plain.load_state_dict(weight)

model_encrypted = Encrypted_MLP(input_dim = 256, output_dim = 10, weight = weight)

# test
def test(model):
  with torch.no_grad(): 
    for images, labels in test_loader:
      outputs = model(images)
      print('Plain model: {}'.format(outputs[0].tolist()))
      return outputs[0].tolist()
      break

result = test(model_plain)

# test
def test2(model):
  with torch.no_grad(): 
    for images, labels in test_loader:
      outputs = model.forward(images)
      outputs = outputs.decrypt().tolist()
      #outputs = torch.tensor(outputs).view(1, -1)
      print('Encrypted model: {}'.format(outputs[0]))
      return outputs[0]
      break

result2 = test2(model_encrypted)

print('Diff(abs): {}'.format([abs(a - b) for a, b in zip(result, result2)]))