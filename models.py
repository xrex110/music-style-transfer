import torch
import torch.nn as nn
import operator
import functools

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
        print(out.shape, x_inp.shape)
        return out


class ESCModel(nn.Module):

    def __init__(self):
        in_shape = (257, 862)
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(257, 2048, kernel_size=2, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size= 2, stride= 2),
            nn.Conv1d(2048, 64, kernel_size=2, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size= 2, stride=2, padding=1)
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *in_shape)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(num_features_before_fcnn, output_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batchsize = x.size(0)
        out = self.feature_extractor(x)
        out = out.view(batchsize, -1)
        out = self.classifier(out)
        return out

class ESCModel2(nn.Module):

    def __init__(self):
        in_shape = (257, 862)
        nn.Module.__init__(self)
            #nn.Conv2d(257, 2048, kernel_size=(2,1), stride=1),
            #nn.LeakyReLU(inplace=True),
            #nn.MaxPool2d(kernel_size= (2,1) stride= 2),
            #nn.Conv2d(2048, 64, kernel_size=(2,1), stride=1),
            #nn.LeakyReLU(inplace=True),
            #nn.MaxPool2d(kernel_size= (2,1), stride=2, padding=1)
        self.conv1 = nn.Conv2d(1, output_channels, kernel_size=(3, 1), stride = 1, padding = 0)
        self.lRelu = nn.LeakyReLU(0.2)
        

        #num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *in_shape)).shape))

        #self.classifier = nn.Sequential(
        #    nn.Linear(num_features_before_fcnn, output_channels),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        '''batchsize = x.size(0)
        out = self.feature_extractor(x)
        #out = out.view(batchsize, -1)
        #out = self.classifier(out)
        return out'''
        out = self.lRelu(self.conv1(x))
        return out

class ESCModel3(nn.Module):

    def __init__(self):
        in_shape = (257, 862, 1)
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(257, 2048, kernel_size=(2,1), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size= (2,1), stride=2),
            nn.Conv2d(2048, 64, kernel_size=(2,1), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *in_shape)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(num_features_before_fcnn, output_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batchsize = x.size(0)
        out = self.feature_extractor(x)
        out = out.view(batchsize, -1)
        out = self.classifier(out)
        return out