import matplotlib.pyplot as plt
from core.dataloader_copy import *
from core.model_copy import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True



dataset = DataLoader('DJI.csv', 0.97, ['Adj Close', 'Volume'], 'Adj Close', True)

train_dt = dataset.get_train_data(51)
test_dt = dataset.get_test_data(51)


# Parameters
dataloader_params = {'batch_size': 1,
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
if torch.cuda.is_available():
    model.cuda()

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)



num_epochs = 1
loss_vals_train = []
loss_vals_test = []
ys_train = []
ys = []


print("Start Training")
for epoch in range(num_epochs):
    print("Epoch nr: "+str(epoch))
    for batch, labels in training_generator:
        batch = batch.view(50,1,-1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        optimiser.zero_grad()

        preds = model(batch)
        ys_train.append(preds.detach())

        loss = loss_fn(preds, labels)
        loss_vals_train.append(loss.item())

        loss.backward()

        optimiser.step()

    for batch, labels in test_generator:
        batch = batch.view(50, 1, -1)
        labels = labels.float()
        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        optimiser.zero_grad()

        y_pred_test = model(batch)
        loss = loss_fn(y_pred_test, labels)
        loss_vals_test.append(loss.item())


lel, y_trainingm8 = dataset.get_train_data(51)

print("Start Predicting")
for epoch in range(num_epochs):
    print("Epoch nr: "+str(epoch))
    model.hidden_1 = model.init_hidden_1()
    model.hidden_2 = model.init_hidden_2()
    for batch, labels in training_generator:
        batch = batch.view(50,1,-1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        preds = model(batch)
        ys.append(preds.detach())

print(ys_train[-1])
print(ys[0])
print(ys[-1])

plt.plot(ys_train, label="Preds during training")
plt.plot(y_trainingm8, label="Data")
plt.legend()
plt.show()
plt.clf()

plt.plot(ys, label="Preds after running training")
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





