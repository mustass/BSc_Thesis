import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

dataset = DataLoader('DJI.csv', 0.77, ['Adj Close', 'Volume'], 'Adj Close', True)

train_dt = dataset.get_train_data(51)
test_dt = dataset.get_test_data(51)


# Parameters
dataloader_params = {'batch_size': 1,
                     'shuffle': True,
                     'drop_last': True,
                     'num_workers': 0}

# Generators
training_set = Dataset(train_dt)

training_generator = data.DataLoader(training_set, **dataloader_params)

test_set = Dataset(test_dt)
test_generator = data.DataLoader(test_set, **dataloader_params)

network_params = {'input_dim': 2,  # As many as there are of columns in data
                  'hidden_dim': 1,
                  'batch_size': dataloader_params['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }

model = Model(**network_params)
if torch.cuda.is_available():
    model.cuda()

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 1
loss_vals_train = []
loss_vals_test = []
ys_train = []
ys_testing = []
ys = []


print("Start Training")
for epoch in range(num_epochs):
    print("Epoch nr: " + str(epoch))
    batch_nr = 0
    for batch, labels in training_generator:
        batch = batch.view(50, 1, -1)
        labels = labels.float()
        #if batch_nr % 1000 == 0:
        #    for name, param in model.named_parameters():
        #        if param.requires_grad:
        #            print(name, param.data)


        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        optimiser.zero_grad()

        preds = model(batch)
        loss = loss_fn(preds, labels)
        if epoch == (num_epochs - 1):
            ys_train.append(preds.detach().cpu().numpy())
            loss_vals_train.append(loss.item())

        loss.backward()

        optimiser.step()
        batch_nr +=1

#    for batch, labels in test_generator:
#        batch = batch.view(50, 1, -1)
#        labels = labels.float()
#        # Transfer to GPU
#        batch, labels = batch.to(device), labels.to(device)
#
#        optimiser.zero_grad()
#
#        y_pred_test = model(batch)
#        loss = loss_fn(y_pred_test, labels)
#        if epoch == (num_epochs - 1):
#            ys_testing.append(y_pred_test.detach().cpu().numpy())
#            loss_vals_test.append(loss.item())



lel, y_trainingm8 = dataset.get_train_data(51)
lel, y_testingm8 = dataset.get_test_data(51)

num_epochs = 1
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


print("Start Predicting")
for epoch in range(num_epochs):
    print("Epoch nr: " + str(epoch))
    model.hidden = model.init_hidden(1)
    # print(model.hidden)
    for batch, labels in training_generator:
        model.eval()
        batch = batch.view(50, 1, -1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        preds = model(batch)
        ys.append(preds.detach().cpu().numpy())
    for batch, labels in test_generator:
        batch = batch.view(50, 1, -1)
        labels = labels.float()
        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        optimiser.zero_grad()

        y_pred_test = model(batch)
        loss = loss_fn(y_pred_test, labels)
        if epoch == (num_epochs - 1):
            ys_testing.append(y_pred_test.detach().cpu().numpy())
            loss_vals_test.append(loss.item())

ys_train = np.array(ys_train)
ys_train = np.reshape(ys_train, (ys_train.shape[0] * ys_train.shape[1], 1))
ys = np.array(ys)
ys = np.reshape(ys, (ys.shape[0] * ys.shape[1], 1))
ys_testing = np.array(ys_testing)
ys_testing = np.reshape(ys_testing, (ys_testing.shape[0] * ys_testing.shape[1], 1))

print(ys_train.shape)
print(ys_train[0])
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

plt.plot(ys_testing, label="Preds during testing")
plt.plot(y_testingm8, label="Data")
plt.legend()
plt.show()
plt.clf()
