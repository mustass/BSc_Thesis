import torch
import numpy as np
import pickle
from torch.backends import cudnn
import time
from ray import tune
from core.model import *
from ray.tune import track
from core.dataloader import *
from core.create_folder import *


def save_checkpoint(state, i, is_best, folder):
    """Save checkpoint if a new best is achieved"""
    filename = folder + '/checkpoint'+str(i)+'.pth.tar'
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    #else:
        #print("=> Accuracy did not improve")


def rolling_window(model_config, datasetName, start_from,
                   window_length, timesteps, max_epochs, folder):
    ### Model:

    model, loss_fn, optimiser = give_me_model(model_config)

    ### CUDA for PyTorch:
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        print("We're running on GPU")
        model.cuda()

    ### Model end

    path = '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/' + datasetName
    dataset = miniDataLoader(path, start_from)

    train_dt = dataset.get_data(timesteps, False, 1)
    predictions = []
    true_labels = []

    end_of_window = 253 * window_length

    end_of_loop = len(train_dt[0])
    print(end_of_loop)
    i = 0
    while end_of_window+i+1 < end_of_loop:
        print(i)
        t0 = time.time()
        train_windows = (train_dt[0][i:end_of_window + i],train_dt[1][i:end_of_window + i])
        test_window = (train_dt[0][end_of_window + i + 1],train_dt[1][end_of_window + i + 1])
        train_windows = Dataset(train_windows)
        dataloader_params = {'batch_size': 1,
                             'shuffle': True,
                             'drop_last': True,
                             'num_workers': 0}
        training_windows_generator = data.DataLoader(train_windows, **dataloader_params)


        avg_loss_epoch = np.zeros((max_epochs, 1))
        best_accuracy = torch.FloatTensor([1000])

        for epoch in range(max_epochs):
            #print("Epoch nr: " + str(epoch))
            loss_vals = np.zeros((max_epochs, len(training_windows_generator)))
            batch_nr = 0
            for batch, labels in training_windows_generator:
                model.batch_size = 1
                batch = batch.view(timesteps, 1, -1)
                # print(batch.shape)
                labels = labels.float()

                # Transfer to GPU
                batch, labels = batch.to(device), labels.to(device)
                optimiser.zero_grad()

                preds = model(batch)
                loss = loss_fn(preds.view(-1), labels)
                loss_vals[epoch, batch_nr] = loss.item()
                loss.backward()
                optimiser.step()

                batch_nr += 1
            avg_loss_epoch[epoch] = np.mean(loss_vals[epoch, :])
            is_best = bool(avg_loss_epoch[epoch] < best_accuracy.numpy())

            best_accuracy = torch.FloatTensor(min(avg_loss_epoch[epoch], best_accuracy.numpy()))
            #print(best_accuracy)
            save_checkpoint({
                'epoch': 0 + epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy
            }, i,is_best, folder)

        path_to_checkpoint = folder + '/' + 'checkpoint' + str(i) + '.pth.tar'
        checkpoint = torch.load(path_to_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        test_seq = torch.from_numpy(test_window[0]).float().view(timesteps, 1, -1).to(device)
        predictions.append(model(test_seq))
        true_labels.append(test_window[1])
        print('The window took {} seconds'.format(time.time() - t0))
        i += 1

    output = (predictions, true_labels)
    file_output = open('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/rolling_window_results/DJI/output'+str(datasetName)+'.obj', 'wb')
    pickle.dump(output, file_output)
    return output

def give_me_model(config):
    network_params = {'input_dim': 1,  # As many as there are of columns in data
                      'hidden_dim': config['hidden_dim'],
                      'batch_size': config['batch_size'],  # From dataloader_parameters
                      'output_dim': 1,
                      'dropout': 0,
                      'num_layers': config['num_layers']
                      }
    model = Model(**network_params)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'])
    return model, loss, optimiser


### Test run:

model_config = {'hidden_dim': 7,
                'batch_size': 1,
                'num_layers': 1,
                'lr': 0.0007255273517151122}

lols = rolling_window(model_config, 'N225.csv', "2012-01-01", 1,29, 50, '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/rolling_window_results/GSPC')