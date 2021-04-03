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

def getPredictedValues(yhat):
	preds = []
	for item in yhat:
		preds.append(np.argmax(item))
	return preds

def accuracy(preds, labels):
	correct = 0.0
	for i in range(len(preds)):
		if preds[i] == labels[i]:
			correct += 1.0
	return (0.0 + correct) / len(preds)

cuda = False
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

print("Cuda", str(cuda))

LOADED_LABELS = False
LOADED_AUDIO = False
LOGGING_INTERVAL = 1

#BATCH_SIZE = 32
BATCH_SIZE = 1

epochs = 3

model = ESCModel3()
if cuda:
	model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)
lossfx = nn.CrossEntropyLoss()

if os.path.exists("train_labels.p"):
	if os.path.getsize("train_labels.p") > 0:
		train_labels_T = pickle.load(open("train_labels.p", "rb"))
		test_labels_T = pickle.load(open("test_labels.p", "rb"))
		LOADED_LABELS = True
		pass

if os.path.exists("train_spectograms.p"):
	if os.path.getsize("train_spectograms.p") > 0:
		print("HERE")
		train_data_T = pickle.load(open("train_spectograms.p", "rb"))
		test_data_T = pickle.load(open("test_spectograms.p", "rb"))
		LOADED_AUDIO = True
		pass

if not LOADED_LABELS:
	print("Building Label Files")
	files = os.listdir("../ESC-50/audio")

	labels = []
	for i in range(0, 2000):
		idx_list = re.split("[0-9]\-[0-9]+\-[A-Z]\-|\.wav", files[i])
		print(f"list: {idx_list}")
		print(f"filenmame: {files[i]}")
		print(f"\tlabel is {idx_list[1]}")

		label_num = int(idx_list[1])

		if label_num < 32:
			labels.append(label_num)

	train_labels = labels[0:1152]
	test_labels = labels[1153:1280]

	train_labels_T = torch.LongTensor(train_labels)
	test_labels_T = torch.LongTensor(test_labels)

	pickle.dump(train_labels_T, open("train_labels.p", "wb"))
	pickle.dump(test_labels_T, open("test_labels.p", "wb"))

	#pickle.dump(labels, open("loaded_labels.p", "wb"))

if not LOADED_AUDIO:
	print("Building Audio Files")
	audio_data = []
	for i in range(0, 2000):
		idx_list = re.split("[0-9]\-[0-9]+\-[A-Z]\-|\.wav", files[i])
		label_num = int(idx_list[1])

		if label_num < 32:
			audio, sr = wav2spectrum("../ESC-50/audio/" + files[i])
			audio_data.append(audio)
			print(f"Loaded {files[i]}")
	
	train_data = audio_data[0:1152]
	test_data = audio_data[1153:1280]

	train_data_T = torch.Tensor(train_data)
	test_data_T = torch.Tensor(test_data)

	pickle.dump(train_data_T, open("train_spectograms.p", "wb"))
	pickle.dump(test_data_T, open("test_spectograms.p", "wb"))

print("Building Datasets")

train_losses=[]

trainDataset = TensorDataset(train_data_T, train_labels_T)
testDataset = TensorDataset(test_data_T, test_labels_T)

train_loader = DataLoader(trainDataset, batch_size = BATCH_SIZE)
test_loader = DataLoader(testDataset, batch_size = BATCH_SIZE)

print("Datasets Built. Beginning Training")

for epoch in range(epochs):
	model.train()
	batch_losses = []
	yhats = []
	ylabs = []

	for index, traindata in enumerate(train_loader):
		x, y = traindata
		ylabs += y
		optimizer.zero_grad()
		if cuda:
			x = x.cuda()
			y = y.cuda()

		yhat = model(x[None, :, :, :])
		loss = lossfx(yhat, y)
		loss.backward()
		batch_losses.append(loss.item())
		optimizer.step()
		yhats.append(np.argmax(yhat.detach().cpu().numpy()[0]))
	train_losses.append(batch_losses)
	print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
	model.eval()
	batch_losses = []
	preds = getPredictedValues(yhats)
	print(f"Accuracy {accuracy(preds, ylabs)}")

torch.save(model.state_dict(), "esc50-model3.pt")