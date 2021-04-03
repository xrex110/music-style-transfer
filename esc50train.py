import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
from models import ESCModel
from utils import *
import pickle

#files = os.listdir("../ESC-50/audio")

#labels = []
#for i in range(0, 2000):
#    idx_list = re.split("[0-9]\-[0-9]+\-[A-E]\-|\.wav", files[i])
    
#    print(f"label is {idx_list[1]}")
#    labels.append(int(idx_list[1]))

#audio_data = []
#for i in range(0, 2000):
#    audio, sr = wav2spectrum(files[i])
#    print(f"Loaded {files[i]}")
#    audio_data.append(audio)

#print(f"type: {type(audio)}")

#pickle.dump(labels, open("loaded_labels.p", "wb"))

audio = pickle.load(open("loaded_spectograms.p", "wb"))
labels = pickle.load(open("loaded_labels.p", "wb"))

train_data = audio[0:1800]
train_labels = labels[0:1800]

test_data = audio[1801:2000]
test_labels = labels[1801:2000]

model = ESCModel()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)
loss_fn = nn.CrossEntropyLoss()

epochs = 60

for epoch in range(epochs):
    model.train()
    


