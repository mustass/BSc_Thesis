import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(pred_labs, true_labs,dates ,folder,title, subtitle,show = True):
    plt.plot(dates,true_labs, label="Actual series")
    plt.plot(dates,pred_labs, label="Predictions")
    plt.legend()
    plt.grid()
    plt.suptitle(t=title, ha = "right")
    plt.title(label=subtitle,ha = "right",fontsize = "medium")

    if folder is not None:
        print("Saving")
        plt.savefig(fname=folder + '/Actual_vs_Preds.png')
    if show:
        plt.show()

    plt.clf()


