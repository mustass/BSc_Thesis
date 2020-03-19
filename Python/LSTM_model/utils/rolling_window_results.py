import pickle
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_rolling_window_results(datatype, result_type, dt):

    path = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/'+datatype+'/'+result_type+'/'+dt

    result = pickle.load(open(path+'/output_file.pickle', "rb" ))

    result_pred = np.array(torch.FloatTensor(result[0]))
    result_labs = np.array(result[1])

    rms = sqrt(mean_squared_error(result_labs, result_pred))
    print(rms)
    with open(path + '/rmse_test_set.txt', 'w+') as f:
        f.write(str(rms))  # Python 3.x
    print("RMSE saved")

    plot_df = pd.DataFrame(
        {'preds': result_pred, 'true_vals': result_labs})
    print(plot_df)
    return plot_df, rms

get_rolling_window_results("returns", "rolling_1", "DJI")
