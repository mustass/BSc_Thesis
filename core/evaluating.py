from core.dataloader import *
from torch.backends import cudnn


def eval_model(trained_model, loss, train_dt, test_dt, dataset, timesteps):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    trained_model = trained_model
    if torch.cuda.is_available():
        trained_model.cuda()
    #######

    loss_fn = loss

    eval_dataloader_params = {'batch_size': 1,
                              'shuffle': False,
                              'drop_last': True,
                              'num_workers': 0}
    trained_model.batch_size = eval_dataloader_params['batch_size']
    # Generators
    training_set = Dataset(train_dt)

    training_generator = data.DataLoader(training_set, **eval_dataloader_params)

    test_set = Dataset(test_dt)
    test_generator = data.DataLoader(test_set, **eval_dataloader_params)

    ys_testing = []
    ys = []
    ys_testing_denormalised = []
    ys__denormalised = []
    loss_vals_train = []
    loss_vals_test = []

    print("Start Evaluating")

    trained_model.hidden = trained_model.init_hidden(1)
    batch_nr = 0
    for batch, labels in training_generator:
        trained_model.eval()
        batch = batch.view(timesteps, 1, -1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        preds = trained_model(batch)
        loss = loss_fn(preds, labels)
        loss_vals_train.append(loss.item())
        ys.append(preds.detach().cpu().numpy())
        ys__denormalised.append(
            denormalise("window", dataset.w_normalisation_p0_train[batch_nr][0], ys[batch_nr], None))
        batch_nr += 1

    batch_nr = 0
    for batch, labels in test_generator:
        batch = batch.view(timesteps, 1, -1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        y_pred_test = trained_model(batch)
        loss = loss_fn(y_pred_test, labels)
        ys_testing.append(y_pred_test.detach().cpu().numpy())
        ys_testing_denormalised.append(
            denormalise("window", dataset.w_normalisation_p0_test[batch_nr][0], ys_testing[batch_nr], None))
        loss_vals_test.append(loss.item())
        batch_nr += 1

    ys = np.array(ys)
    ys = np.reshape(ys, (ys.shape[0] * ys.shape[1], 1))
    ys__denormalised = np.array(ys__denormalised)
    ys__denormalised = np.reshape(ys__denormalised, (ys__denormalised.shape[0] * ys__denormalised.shape[1], 1))
    ys_testing_denormalised = np.array(ys_testing_denormalised)
    ys_testing_denormalised = np.reshape(ys_testing_denormalised,
                                         (ys_testing_denormalised.shape[0] * ys_testing_denormalised.shape[1], 1))
    ys_testing = np.array(ys_testing)
    ys_testing = np.reshape(ys_testing, (ys_testing.shape[0] * ys_testing.shape[1], 1))

    return ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train
