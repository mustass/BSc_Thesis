import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

path = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/rolling_window_results/DJI'

result = pickle.load( open( '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/rolling_window_results/DJI/outputDJI.csv.obj', "rb" ))

result_pred = np.array(torch.FloatTensor(result[0]))
result_labs = np.array(result[1])


plt.plot(result_labs, label="Actual data")
plt.plot(result_pred, label="Predictions")
plt.legend()
plt.savefig(fname=path + '/rolling_window_result' + '.png')
plt.show()

rms = sqrt(mean_squared_error(result_labs, result_pred))
print(rms)