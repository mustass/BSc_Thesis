import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core.dataloader_copy import *
from core.model_copy import *

dataset = DataLoader('DJI.csv', 0.8, ['Adj Close', 'Volume'], 'Adj Close', True)

train_dt = dataset.get_train_data(51)
test_dt = dataset.get_test_data(51)

# Parameters
dataloader_params = {'batch_size': 25,
                     'shuffle': False,
                     'drop_last': True,
                     'num_workers': 0}



# Generators
training_set = Dataset(train_dt)

training_generator = data.DataLoader(training_set, **dataloader_params)

test_set = Dataset(test_dt)
test_generator = data.DataLoader(test_set, **dataloader_params)


network_params = {'input_dim': 2,  # As many as there are of columns in data
                  'hidden_dim': 100,
                  'batch_size': dataloader_params['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0.2,
                  'num_layers': 2,
                  'hidden_dim_2': 50,
                  'dropout_2': 0.2,
                  'btwn_lyr_dropout': 0.3,
                  'num_layers_2': 2,
                  }

model = Model(**network_params)
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1
loss_vals_train = []
loss_vals_test = []
ys = []
for epoch in range(num_epochs):
    for batch, labels in training_generator:
        batch = batch.view(50,1,-1)
        labels = labels.float()
        optimiser.zero_grad()

        y_pred_train = model(batch)
        #ys.append(y_pred_train.detach())

        loss = loss_fn(y_pred_train, labels)
        loss_vals_train.append(loss.item())

        loss.backward()

        optimiser.step()

    for batch, labels in test_generator:
        batch = batch.view(50, 1, -1)
        labels = labels.float()
        optimiser.zero_grad()

        y_pred_test = model(batch)
        #print("Prediction:")
        #print(y_pred_test)

        loss = loss_fn(y_pred_test, labels)
        loss_vals_test.append(loss.item())
        # print("LOSS")
        #print(loss)

training_set, y_trainingm8 = dataset.get_train_data(30)


# Parameters
dataloader_params = {'batch_size': 6938,
                     'shuffle': False,
                     'drop_last': True,
                     'num_workers': 0}



# Generators
training_set = Dataset(train_dt)
training_generator = data.DataLoader(training_set, **dataloader_params)
print(len(training_generator))
test_set = Dataset(test_dt)
test_generator = data.DataLoader(test_set, **dataloader_params)

model.batch_size = 6938
num_epochs = 1
for epoch in range(num_epochs):
    print("lol")
    for batch, labels in training_generator:
        print("lolsies1")
        batch = batch.view(50,6938,-1)
        print("lolsies2")
        labels = labels.float()
        optimiser.zero_grad()
        print("lolsies3")
        y_pred_train = model(batch)
        print("lolsies4")
        ys.append(y_pred_train.detach())

        loss = loss_fn(y_pred_train, labels)





plt.plot(ys, label="Preds")
plt.plot(y_trainingm8, label="Data")
plt.legend()
plt.show()
plt.clf()


plt.plot(loss_vals_train, label="Training loss")
plt.legend()
plt.show()
plt.clf()

plt.plot(loss_vals_test, label="Test loss")
plt.legend()
plt.show()
plt.clf()





