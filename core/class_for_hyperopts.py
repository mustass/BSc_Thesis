import torch
import numpy as np
from torch.backends import cudnn
import time
from ray import tune
from core.model import *
from ray.tune import track
from core.dataloader import *
from core.create_folder import *

class training_instance():
    '''A class for creating a training instance for a dataset'''

    def __init__(self, path, split, cols, label_col, MinMax):
        self.path_to_data = path
        self.train_test_split = split
        self.feature_cols = cols
        self.label_col = label_col
        self.MinMax_normalisation = MinMax

        self.dataset = DataLoader(path=path, split=split,
                         cols=['Adj Close', 'Volume'],
                         label_col='Adj Close', MinMax=False)


















def train_hypopt(config):
    dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv', split=0.80,
                         cols=['Adj Close', 'Volume'],
                         label_col='Adj Close', MinMax=False)

    timesteps = config["timesteps"]
    train_dt = dataset.get_train_data(timesteps, True)

    test_dt = dataset.get_test_data(timesteps, True)

    # Parameters
    dataloader_params_train = {'batch_size': 1,
                               'shuffle': True,
                               'drop_last': True,
                               'num_workers': 0}
    # Parameters
    dataloader_params_test = {'batch_size': 1,
                              'shuffle': False,
                              'drop_last': True,
                              'num_workers': 0}
    # Generators
    training_set = Dataset(train_dt)
    training_generator = data.DataLoader(training_set, **dataloader_params_train)
    test_set = Dataset(test_dt)
    test_generator = data.DataLoader(test_set, **dataloader_params_test)
    epochs = 150
    ### Saving:
    folder_name = 'Run_w_' + str(config["timesteps"]) + 'timesteps_' + str(config["hidden_dim"]) + 'hiddenDim' + str(
        config["num_layers"]) + 'layers'
    new_folder = create_folder('/home/s/Dropbox/KU/BSc Stas/Python/Try_again' + '/results', folder_name)

    # Model:
    network_params = {'input_dim': 2,
                      'hidden_dim': config["hidden_dim"],
                      'batch_size': 1,
                      'output_dim': 1,
                      'dropout': 0,
                      'num_layers': config["num_layers"]
                      }
    model = Model(**network_params)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    scheduler = None

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        # print("We're running on GPU")
        model.cuda()
    #######

    lwst_error = 1000
    while True:
        error, model = one_epoch_training(model, loss, optimiser,
                                          scheduler, device,
                                          training_generator,
                                          test_generator, timesteps,
                                          1)

        is_best = bool(error < lwst_error)
        print(
            "error of the epoch: " + str(error) + " best accuracy before : " + str(lwst_error))
        print("Best accuracy currently is: " + str(min(error, lwst_error)))
        lwst_error = min(error, lwst_error)

        save_checkpoint({
            'epoch': 'tuning',
            'state_dict': model.state_dict(),
            'best_accuracy': lwst_error
        }, is_best, new_folder)

        track.log(error=-error)


def one_epoch_training(model, loss, optimiser, scheduler, device, train_gen, test_gen,
                       timesteps, batch_size):
    t0 = time.time()
    scheduler = scheduler
    optimiser = optimiser
    loss_fn = loss

    # Track losses:
    loss_vals_train = np.zeros((len(train_gen)))
    loss_vals_test = np.zeros((len(test_gen)))

    total_time = 0
    batch_nr = 0
    for batch, labels in train_gen:
        model.batch_size = batch_size
        batch = batch.view(timesteps, batch_size, -1)
        # print(batch.shape)
        labels = labels.float()
        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)
        optimiser.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, labels)
        loss_vals_train[batch_nr] = loss.item()
        loss.backward()
        optimiser.step()

        if scheduler is not None:
            scheduler.step()
        batch_nr += 1

        batch_nr = 0
    for batch, labels in test_gen:
        model.batch_size = 1
        batch = batch.view(timesteps, 1, -1)
        labels = labels.float()
        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)
        optimiser.zero_grad()
        y_pred_test = model(batch)
        loss = loss_fn(y_pred_test, labels)
        loss_vals_test[batch_nr] = loss.item()
        batch_nr += 1

    error = np.mean(loss_vals_test)
    print('The epoch took {} seconds'.format(time.time() - t0))
    total_time += time.time() - t0

    return error, model


def save_checkpoint(state, is_best, folder):
    """Save checkpoint if a new best is achieved"""
    filename = folder + '/checkpoint.pth.tar'
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")
