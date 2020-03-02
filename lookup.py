"""
File for evaluating and working with a previously trained model.
"""
from ray.tune import Analysis
import torch
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
configs = []
analysis_GSPC = Analysis("~/ray_results/1forward_returns_GSPC")
analysis_IXIC = Analysis("~/ray_results/1forward_returns_IXIC")
analysis_N225 = Analysis("~/ray_results/1forward_returns_N225")
analysis_DJI  = Analysis("~/ray_results/1forward_returns_DJI")
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




path_to_folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/Predicting1_w_29_timesteps_7_hiddenDim_1_layers_0.0007255273517151122_LR'
path_to_checkpoint = path_to_folder + '/' + 'checkpoint' + '.pth.tar'
path_to_dataset = '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/IXIC.csv'

cuda = torch.cuda.is_available()
if cuda:
    checkpoint = torch.load(path_to_checkpoint)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(path_to_checkpoint,
                            map_location=lambda storage,
                                                loc: storage)

print("Best accuracy, aka. lowest error is: " + str(checkpoint['best_accuracy']))


dataset = DataLoader(path_to_dataset, 0.80, ['log_ret'],
                     'log_ret', False)
timesteps = 29
num_forward = 1

network_params = {'input_dim': 1,  # As many as there are of columns in data
                  'hidden_dim': 7,
                  'batch_size': 1,  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }

Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_model.load_state_dict(checkpoint['state_dict'])
ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train = eval_model(Nice_model,
                                                                                                        Nice_loss,
                                                                                                        dataset,
                                                                                                        timesteps,
                                                                                                        num_forward)
train_dt = dataset.get_train_data(timesteps, False, num_forward)
test_dt = dataset.get_test_data(timesteps, False, num_forward)
print(np.sqrt(np.mean(loss_vals_test)))
y_training = train_dt[1]
y_testing = test_dt[1]

plot_and_save(ys, ys_testing, y_training, y_testing, None, None,
              loss_vals_train,
              loss_vals_test, True, "Checkpoint_model",
              path_to_folder + '/')
#sequences, sequences_denormalized = predict_seq_avg(Nice_model, dataset, timesteps, 25)

test_dt = dataset.get_test_data(timesteps, False)
y_testing = test_dt[1]
#plot_results_multiple(sequences_denormalized, y_testing, 25,
#                      path_to_folder + '/')
