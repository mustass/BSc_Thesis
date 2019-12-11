import torch
import numpy as np
from torch.backends import cudnn


def train_model(model, loss, optimiser, epochs, train_gen, test_gen,
                timesteps, batch_size):
    ######## CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        model.cuda()
    #######

    optimiser = optimiser
    loss_fn = loss

    # Track smthngs:

    loss_vals_train = np.zeros((epochs, len(train_gen)))
    loss_vals_test = np.zeros((epochs, len(test_gen)))

    print("Start Training")

    for epoch in range(epochs):
        print("Epoch nr: " + str(epoch))
        batch_nr = 0
        for batch, labels in train_gen:
            batch = batch.view(timesteps, batch_size, -1)
            labels = labels.float()

            # if batch_nr % 1000 == 0:
            #    for name, param in model.named_parameters():
            #        if param.requires_grad:
            #            print(name, param.data)

            # Transfer to GPU
            batch, labels = batch.to(device), labels.to(device)

            optimiser.zero_grad()

            preds = model(batch)
            loss = loss_fn(preds, labels)
            loss_vals_train[epoch, batch_nr] = loss.item()

            loss.backward()

            optimiser.step()
            batch_nr += 1

        batch_nr = 0
        for batch, labels in test_gen:
            batch = batch.view(timesteps, batch_size, -1)
            labels = labels.float()
            # Transfer to GPU
            batch, labels = batch.to(device), labels.to(device)

            optimiser.zero_grad()

            y_pred_test = model(batch)
            loss = loss_fn(y_pred_test, labels)
            loss_vals_test[epoch, batch_nr] = loss.item()
            batch_nr += 1

    return model, loss_vals_train, loss_vals_test
