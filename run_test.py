import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *

dataset = DataLoader('DJI.csv', 0.77, ['Adj Close', 'Volume'], 'Adj Close', True)
timesteps = 50
train_dt = dataset.get_train_data(timesteps)
test_dt = dataset.get_test_data(timesteps)


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
                  'hidden_dim': 10,
                  'batch_size': dataloader_params['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }
epochs = 25
Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_optimiser = torch.optim.Adam(Nice_model.parameters(), lr=0.005)
Nice_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Nice_optimiser,epochs)

Nice_model_trained, loss_train, loss_test = train_model(Nice_model, Nice_loss, Nice_optimiser,Nice_scheduler ,epochs,training_generator,
                                                       test_generator, timesteps, dataloader_params['batch_size'])

ys, ys_testing, loss_vals_test,loss_vals_train = eval_model(Nice_model_trained,Nice_loss, train_dt,test_dt,timesteps)

y_trainingm8= train_dt[1]
y_testingm8 = test_dt[1]

plot_and_save(ys,ys_testing,y_trainingm8, y_testingm8, loss_vals_train, loss_vals_test, False, Nice_model_trained,
              'Nice_model_with_LRANNEAL_hidden_dim10_batchsize1')

