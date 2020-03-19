import torch
import numpy as np
from torch.backends import cudnn
import time
from ray import tune
from core.model import *
from ray.tune import track
from core.dataloader import *
from core.create_folder import *


def train_model(model, loss, optimiser, scheduler, max_epochs, train_gen, test_gen,
                timesteps, batch_size, folder, resuming):
    ######## CUDA for PyTorch
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        print("We're running on GPU")
        model.cuda()
    #######
    scheduler = scheduler
    optimiser = optimiser
    loss_fn = loss

    # Track losses:

    loss_vals_train = np.zeros((max_epochs, len(train_gen)))
    loss_vals_test = np.zeros((max_epochs, len(test_gen)))

    avg_loss_epoch_train = np.zeros((max_epochs, 1))
    avg_loss_epoch_test = np.zeros((max_epochs, 1))

    print("Start Training")
    total_time = 0
    best_accuracy = torch.FloatTensor([1000])

    start_epoch = 0
    if resuming:
        path_to_checkpoint = folder + '/checkpoint.pth.tar'
        cuda = torch.cuda.is_available()
        if cuda:
            checkpoint = torch.load(path_to_checkpoint)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(path_to_checkpoint,
                                    map_location=lambda storage,
                                                        loc: storage)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(path_to_checkpoint, checkpoint['epoch']))

    for epoch in range(start_epoch, max_epochs, 1):
        print("Epoch nr: " + str(epoch))
        t0 = time.time()
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
            loss_vals_train[epoch, batch_nr] = loss.item()

            loss.backward()

            optimiser.step()

            batch_nr += 1
        if scheduler is not None:
            scheduler.step()

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
            loss_vals_test[epoch, batch_nr] = loss.item()
            batch_nr += 1

        error = np.mean(loss_vals_test[epoch, :])
        print('The epoch took {} seconds'.format(time.time() - t0))
        total_time += time.time() - t0

        avg_loss_epoch_train[epoch] = np.mean(loss_vals_train[epoch, :])
        avg_loss_epoch_test[epoch] = np.mean(loss_vals_test[epoch, :])

        is_best = bool(avg_loss_epoch_test[epoch] < best_accuracy.numpy())

        best_accuracy = torch.FloatTensor(min(avg_loss_epoch_test[epoch], best_accuracy.numpy()))
        print(best_accuracy)
        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best, folder)

    print('Total training time is: ' + str(total_time) + ' seconds')
    save_path = folder + '/' + 'last_model.pth.tar'
    torch.save({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, save_path)

    with open(folder + '/model_summary.txt', 'w+') as f:
        f.write(str(model))  # Python 3.x
    print("Last Model Saved")
    return model, avg_loss_epoch_train, avg_loss_epoch_test, error


def save_checkpoint(state, is_best, folder):
    """Save checkpoint if a new best is achieved"""
    filename = folder + '/checkpoint.pth.tar'
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def train_hypopt(config):
    dataset = DataLoader(path=config["filename"], split=0.80,
                         cols=['log_ret'], start_from= "1985-01-01", end = "1995-01-01",
                         label_col='log_ret', MinMax=False)

    timesteps = config["timesteps"]
    train_dt = dataset.get_train_data(timesteps, config["window_normalisation"], config["num_forward"])

    test_dt = dataset.get_test_data(timesteps, config["window_normalisation"], config["num_forward"])
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
    ### Saving:
    folder_name = str(config["num_forward"])+'_forward_usng_' + str(config["timesteps"]) + '_timesteps_' + str(config["hidden_dim"]) + '_hiddenDim_' + str(
        config["num_layers"]) + '_layers_'+str(config["lr"]) + "_LR"
    new_folder = create_folder(config["path"], folder_name)

    # Model:
    network_params = {'input_dim': 1,
                      'hidden_dim': config["hidden_dim"],
                      'batch_size': 1,
                      'output_dim': 1,
                      'dropout': config["dropout"],
                      'num_layers': config["num_layers"]
                      }
    model = Model(**network_params)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
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
