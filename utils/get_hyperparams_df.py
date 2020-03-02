"""
File for looking through hyperopt results and finding the best hyperparams for the backtests.
"""
from ray.tune import Analysis
from core.pred_sequence import *
configs = []
analysis_GSPC = Analysis("~/ray_results/1forward_returns_GSPC")
analysis_IXIC = Analysis("~/ray_results/1forward_returns_IXIC")
analysis_N225 = Analysis("~/ray_results/1forward_returns_N225")
analysis_DJI  = Analysis("~/ray_results/Jan21")
configs.append(analysis_GSPC.get_best_config(metric="error", mode="max"))
configs.append(analysis_IXIC.get_best_config(metric="error", mode="max"))
configs.append(analysis_N225.get_best_config(metric="error", mode="max"))
configs.append(analysis_DJI.get_best_config(metric="error", mode="max"))
configs_np = np.empty(shape=(4, 5))

for i in range(4):
    configs_np[i][0] = configs[i]["num_layers"]
    configs_np[i][1] = configs[i]["hidden_dim"]
    configs_np[i][2] = configs[i]["lr"]
    configs_np[i][3] = configs[i]["timesteps"]
    configs_np[i][4] = 0.2
        #configs[i]["dropout"]
configs_df = pd.DataFrame(configs_np, index = ["SP500","IXIC","Nikkei 225", "DJI"], columns=["Num. layers", "Hidden dim", "LR", "Timesteps", "Dropout"])
print(configs_df)