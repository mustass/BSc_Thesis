import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_and_save(pred_train, pred_test, label_train, label_test,
                  train_error_mtrx, test_error_mtrx, train_error, test_error,
                  show,model,folder):
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
        plt.savefig(fname=folder + '/Train_vs_actual_'+str(model)+'.png')

    plt.clf()

    plt.plot(train_error, label="Loss for every sequence in Training Set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Train_loss_'+str(model)+'.png')
    plt.clf()

    plt.plot(test_error, label="Loss for every sequence in Test Set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Test_loss_'+str(model)+'.png')
    plt.clf()

    plt.plot(pred_test, label="Predictions on test set")
    plt.plot(label_test, label="Actual data")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/Test_vs_actual_'+str(model)+'.png')
    plt.clf()

    plt.plot(A, label="Avg loss per epoch on train set")
    plt.plot(B, label="Avg loss per epoch on test set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/losses_per_epoch_'+str(model)+'.png')
    plt.clf()

    plt.plot(A, label="Avg loss per epoch on train set")
    plt.legend()
    if show:
        plt.show()
    if folder is not None:
        plt.savefig(fname=folder + '/losses_per_epoch_only_train_'+str(model)+'.png')
    plt.clf()