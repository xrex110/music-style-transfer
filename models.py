import torch
import torch.nn as nn

cuda = False
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

num_fft = 512
num_channels = round(1 + num_fft/2)
output_channels = 32

class MusicCNN(nn.Module):
    def __init__(self):
        super(MusicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, output_channels, kernel_size=(3, 1), stride = 1, padding = 0)
        self.lRelu = nn.LeakyReLU(0.2)

        #set rand weights
        weights = torch.randn(self.conv1.weight.data.shape)
        self.conv1.weight = torch.nn.Parameter(weights, requires_grad=False)
        bias = torch.zeros(self.conv1.bias.data.shape)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_inp):
        out = self.lRelu(self.conv1(x_inp))
        return out

class ESCModel(nn.Module):
    def __init__(self):
        super(ESCModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 2048, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(2048, 64, kernel_size=2, stride=2)
        self.linear = nn.Linear(64, 50)  #Paper uses 32 ???

        self.lRelu = nn.LeakyReLU()

    def forward(self, inp):
        out1 = self.lRelu(self.conv1(inp))
        out2 = self.lRelu(self.conv2(out1))

        out = self.linear(out2)

        return out