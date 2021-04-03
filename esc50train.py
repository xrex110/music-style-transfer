import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
from models import ESCModel
from utils import *
import pickle
import time

from torch.utils.data import TensorDataset, DataLoader

cuda = False
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

print("Cuda", str(cuda))

LOADED = False
LOGGING_INTERVAL = 1

BATCH_SIZE = 32

epochs = 60

if os.path.exists("loaded_labels.p") and os.path.exists("loaded_spectograms.p"):
	if os.path.getsize("loaded_labels.p") > 0 and os.path.getsize("loaded_spectograms.p") > 0:
		# Comment this line out to force it to reload the files
		LOADED = True
		pass

if not LOADED:

	files = os.listdir("../ESC-50/audio")

	labels = []
	for i in range(0, 2000):
		idx_list = re.split("[0-9]\-[0-9]+\-[A-Z]\-|\.wav", files[i])
		print(f"list: {idx_list}")
		print(f"filenmame: {files[i]}")
		print(f"\tlabel is {idx_list[1]}")
		vect = np.zeros([50])
		vect[int(idx_list[1])] = 1.0
		labels.append(vect)

	audio_data = []
	for i in range(0, 2000):
		audio, sr = wav2spectrum("../ESC-50/audio/" + files[i])
		print(f"Loaded {files[i]}")
		audio_data.append(audio)

	print(f"type: {type(audio)}")

	pickle.dump(labels, open("loaded_labels.p", "wb"))
	pickle.dump(audio_data, open("loaded_spectograms.p", "wb"))

else:
	with open("loaded_spectograms.p", "rb") as file:
		audio = pickle.load(file)
	with open("loaded_labels.p", "rb") as file:
		labels = pickle.load(file)

train_data = audio[0:1800]
train_labels = labels[0:1800]

test_data = audio[1801:2000]
test_labels = labels[1801:2000]

model = ESCModel()
if cuda:
	model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)
lossfx = nn.CrossEntropyLoss()

train_losses=[]

train_data_T = torch.Tensor(train_data)
train_labels_T = torch.Tensor(train_labels)
test_data_T = torch.Tensor(test_data)
test_labels_T = torch.Tensor(test_labels)

trainDataset = TensorDataset(train_data_T, train_labels_T)
testDataset = TensorDataset(test_data_T, test_labels_T)

train_loader = DataLoader(trainDataset, batch_size = BATCH_SIZE)
test_loader = DataLoader(testDataset, batch_size = BATCH_SIZE)

for epoch in range(epochs):
	model.train()
	batch_losses=[]

	for index, traindata in enumerate(train_loader):
		x, y = traindata
		optimizer.zero_grad()
		if cuda:
			x = x.cuda()
			y = y.cuda()

		yhat = model(x)
		loss = lossfx(yhat, y)
		loss.backward()
		batch_losses.append(loss.item())
		optimizer.step()
	train_losses.append(batch_losses)
	print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
	model.eval()
	batch_losses = []
	#trace_y = []
	#trace_yhat = []


'''
for epoch in range(epochs):
	model.train()
	x, y = train_data, train_labels
	optimizer.zero_grad()
	print(type(train_data[0]), type(train_labels[0]))
	x = x.to(device, dtype=torch.float32)
	y = y.to(device, dtype=torch.long)
	yhat = model(x)
	loss = loss_fn(y_hat, y)
	loss.backward()
	train_losses.append(loss.item())
	optimizer.step()
	acc = accuracy(getPredictedValues(yhat), y)
	print(f"Epoch {epoch} Training Loss {loss.item()} Training Accuracy {acc}")
'''

def getPredictedValues(yhat):
	preds = []
	for item in yhat:
		ind = np.argmax(item)
		new = np.zeros([50])
		new[ind] = 1
		preds.append(new)

def accuracy(preds, labels):
	correct = 0.0
	for i in range(len(preds)):
		if preds[i] == labels[i]:
			correct += 1.0
	return (0.0 + correct) / len(preds)