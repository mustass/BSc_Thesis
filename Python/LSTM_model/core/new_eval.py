from core.dataloader import *
from torch.backends import cudnn


def eval_model(trained_model, loss, dataset, timesteps):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    trained_model = trained_model
    if torch.cuda.is_available():
        trained_model.cuda()
    #######


    loss_fn = loss

    test_generator = dataset

    ys_testing = []
    loss_vals_test = []

    print("Start Evaluating")

    batch_nr = 0
    for batch, labels in test_generator:
        batch = batch.view(timesteps, 1, -1)
        labels = labels.float()

        # Transfer to GPU
        batch, labels = batch.to(device), labels.to(device)

        y_pred_test = trained_model(batch)
        loss = loss_fn(y_pred_test, labels)
        ys_testing.append(y_pred_test.detach().cpu().numpy())
        loss_vals_test.append(loss.item())
        batch_nr += 1

    ys_testing = np.array(ys_testing)
    ys_testing = np.reshape(ys_testing, (ys_testing.shape[0] * ys_testing.shape[1], 1))
    loss_vals_test = np.array(loss_vals_test)
    RMSE = np.sqrt(np.mean(loss_vals_test))

    return ys_testing, loss_vals_test, RMSE