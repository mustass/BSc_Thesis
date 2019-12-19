import torch
import numpy as np
from torch.backends import cudnn
import time


def train_model(model, loss, optimiser, scheduler, max_epochs, train_gen, test_gen,
                timesteps, batch_size, folder,resuming):
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
        path_to_checkpoint = folder+'/checkpoint.pth.tar'
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


    for epoch in range(start_epoch, max_epochs,1):
        print("Epoch nr: " + str(epoch))
        t0 = time.time()
        batch_nr = 0
        for batch, labels in train_gen:
            model.batch_size = batch_size
            batch = batch.view(timesteps, batch_size, -1)
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
        print('The epoch took {} seconds'.format(time.time() - t0))
        total_time += time.time() - t0

        avg_loss_epoch_train[epoch] = np.mean(loss_vals_train[epoch, :])
        avg_loss_epoch_test[epoch] = np.mean(loss_vals_test[epoch, :])

        is_best = bool(avg_loss_epoch_test[epoch] < best_accuracy.numpy())

        best_accuracy = torch.FloatTensor(min(avg_loss_epoch_test[epoch], best_accuracy.numpy()))

        save_checkpoint({
            'epoch': start_epoch + epoch+1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best, folder)

    print('Total training time is: ' + str(total_time)+' seconds')
    return model, loss_vals_train, loss_vals_test


def save_checkpoint(state, is_best, folder):
    """Save checkpoint if a new best is achieved"""
    filename = folder + '/checkpoint.pth.tar'
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")