import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_and_save(pred_train, pred_test, label_train, label_test,
                  train_error_mtrx, test_error_mtrx, train_error, test_error,
                  show, trained_model, folder):
    save_path = folder + '/' + 'model.pt'
    torch.save(trained_model.state_dict(), save_path)
    with open(folder + '/model_summary.txt', 'w+') as f:
        f.write(str(trained_model))  # Python 3.x
    print("Model Saved")

    if train_error_mtrx is not None:
        A = np.mean(train_error_mtrx, axis=1)
    if test_error_mtrx is not None:
        B = np.mean(test_error_mtrx, axis=1)

    plt.plot(pred_train, label="Predictions on train set")
    plt.plot(label_train, label="Actual data")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Train_vs_actual.png')

    plt.clf()

    plt.plot(train_error, label="Loss for every sequence in Training Set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Train_loss.png')
    plt.clf()

    plt.plot(test_error, label="Loss for every sequence in Test Set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Test_loss.png')
    plt.clf()

    plt.plot(pred_test, label="Predictions on test set")
    plt.plot(label_test, label="Actual data")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Test_vs_actual.png')
    plt.clf()

    plt.plot(A, label="Avg loss per epoch on train set")
    plt.plot(B, label="Avg loss per epoch on test set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/losses_per_epoch.png')
    plt.clf()

    plt.plot(A, label="Avg loss per epoch on train set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/losses_per_epoch_only_train.png')
    plt.clf()