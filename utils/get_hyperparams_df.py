"""
File for looking through hyperopt results and finding the best hyperparams for the backtests.
"""
import pickle
from ray.tune import Analysis
from core.pred_sequence import *
configs = []

configs.append({'filename': '/content/drive/My Drive/BSc_work/data/DJI.csv', 'path': '/content/drive/My Drive/BSc_work/bsc_lstm/results/hyperopt/DJI', 'window_normalisation': False, 'num_forward': 1, 'dropout': 0.12924616463481806, 'hidden_dim': 13, 'lr': 0.00032523707257407953, 'num_layers': 3, 'timesteps': 38})
configs.append({'filename': '/content/drive/My Drive/BSc_work/data/IXIC.csv', 'path': '/content/drive/My Drive/BSc_work/bsc_lstm/results/hyperopt/IXIC', 'window_normalisation': False, 'num_forward': 1, 'dropout': 0.4049558310874882, 'hidden_dim': 2, 'lr': 0.019981598297026913, 'num_layers': 1, 'timesteps': 39})
configs.append({'filename': '/content/drive/My Drive/BSc_work/data/N225.csv', 'path': '/content/drive/My Drive/BSc_work/bsc_lstm/results/hyperopt/N225', 'window_normalisation': False, 'num_forward': 1, 'dropout': 0.3936630534624274, 'hidden_dim': 14, 'lr': 0.07054668817815411, 'num_layers': 4, 'timesteps': 5})
configs.append({'filename': '/content/drive/My Drive/BSc_work/data/GSPC.csv', 'path': '/content/drive/My Drive/BSc_work/bsc_lstm/results/hyperopt/GSPC', 'window_normalisation': False, 'num_forward': 1, 'dropout': 0.2541310946411278, 'hidden_dim': 6, 'lr': 0.003605592052286579, 'num_layers': 3, 'timesteps': 36})

configs_np = np.empty(shape=(4, 5))
datasets = ["DJI", "IXIC", "N225", "GSPC"]
for i in range(4):
    configs_np[i][1-1] = configs[i]["num_layers"]
    configs_np[i][2-1] = configs[i]["hidden_dim"]
    configs_np[i][3-1] = configs[i]["lr"]
    configs_np[i][4-1] = configs[i]["timesteps"]
    configs_np[i][5-1] = configs[i]["dropout"]
configs_df = pd.DataFrame(configs_np,
                          index = [ "DJI","IXIC","N225","GSPC"],
                          columns=["Num. layers", "Hidden dim", "LR", "Timesteps", "Dropout"])
file_output = open('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/hyperopt/best_configs_dataframe.pickle', 'wb')
pickle.dump(configs_df, file_output)
print(configs_df)