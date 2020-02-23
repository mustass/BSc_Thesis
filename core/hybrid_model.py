"""
This file is used to create the hybrid model.
The hybrid model is constructed of *n* models, to predict *n* steps forward.
The *n* models are each trained to predict:
1: 1 forward
2: 2 forward
.....
n: n forward

Arguments:
-- The 20 models' state dictionaries and configurations.
"""

# config = {"1forward": {"hidden_dim", "num_layers", "timesteps", "state_dict", "num_forward"},
#         "2forward": {"hidden_dim", "num_layers", "timesteps", "state_dict", "num_forward"},
#         "3forward": {"hidden_dim", "num_layers", "timesteps", "state_dict", "num_forward"},
#         "4forward": {"hidden_dim", "num_layers", "timesteps", "state_dict", "num_forward"},
#         }

from core.model import *
from core.dataloader import *
import numpy as np
import torch


class hybrid_model():
    def __init__(self, num_models, model_keys, config):

        # Saving info:
        self.num_models = num_models
        self.model_config = config
        self.model_keys = model_keys
        # Create a list to track the timesteps used by the models:
        self.list_of_timesteps = []
        # Create an empty dictionary to store compiled model objects:
        self.models_dict = {}
        # Compile the models:
        for model in range(num_models):
            # Create model key:
            #print(model)
            model_key = self.model_keys[model]
            # Network parameters:
            print(config[model_key])
            network_params = {'input_dim': 2,
                              'hidden_dim': config[model_key]["hidden_dim"],
                              'batch_size': 1,
                              'output_dim': 1,
                              'dropout': 0,
                              'num_layers': config[model_key]["num_layers"]
                              }
            # Compile model and recreate from checkpoint:
            model = Model(**network_params)
            #print(config[model_key]['state_dict'])
            model.load_state_dict(config[model_key]['state_dict'])
            # Add model to the model_list dictionary:
            self.models_dict[model_key] = model
            # Track how many timesteps they use:
            self.list_of_timesteps.append(config[model_key]["timesteps"])
        # Save prediction length:
        self.prediction_length = None
        # Save the most timesteps used by any of the models:
        self.most_timesteps = max(self.list_of_timesteps)
        # Save the predictions:
        self.number_of_predictions_list = []
        self.preds_dic = {}
        self.denorm_preds_dic = {}

    def _predict_n_forward(self, dataset, model_key, window_normalisation=True):
        """
        Idea:
            Do a dataset with the longest tiemesteps
            feed it to run_predictions
            inside run predictions split it up to accommodate shorter timestep sequences
         """
        timesteps = self.model_config[model_key]["timesteps"]
        model = self.models_dict[model_key]
        n_forward = self.model_config[model_key]['num_forward']
        dt = dataset.get_test_data(timesteps, window_normalisation, n_forward)[0]
        predicted = []
        predicted_denormalized = []
        skip = self.most_timesteps - timesteps

        for i in range(int(len(dt) / self.prediction_length)):
            if skip + i * self.prediction_length < dt.shape[0]:
                curr_frame = dt[skip + i * self.prediction_length]
                curr_frame = torch.from_numpy(curr_frame).type(torch.Tensor).detach().view(timesteps, 1, -1)
                prediction = model(curr_frame)[0, 0].detach()
                predicted.append(prediction.cpu().numpy())
                predicted_denormalized.append(
                    denormalise("window", dataset.w_normalisation_p0_test[i * self.prediction_length][0],
                                prediction.cpu().numpy(), None)
                )
        return predicted, predicted_denormalized

    def run_predictions(self, dataset, prediction_length, window_normalisation=True):
        """
        Loops over _predict()
        """
        assert (prediction_length <= self.num_models)
        self.prediction_length = prediction_length
        for i in range(prediction_length):
            model_key = self.model_keys[i]
            print(model_key)
            preds, denorm_preds = self._predict_n_forward(dataset, model_key, window_normalisation)
            self.number_of_predictions_list.append(len(denorm_preds))
            print(len(denorm_preds))
            self.preds_dic[model_key] = preds
            self.denorm_preds_dic[model_key] = denorm_preds

    def get_predictions(self, mode='denormalized'):
        """
        This function will put predictions in order into array of shape:
        (num_predictions, prediction_len)
        """
        print(self.number_of_predictions_list)
        assert (mode in ['denormalized', 'normalized'])
        if mode == 'denormalized':
            predictions_dict = self.denorm_preds_dic
        else:
            predictions_dict = self.preds_dic
        predictions_array = np.empty((max(self.number_of_predictions_list), self.prediction_length))

        for i in range(predictions_array.shape[0]):
            for j in range(predictions_array.shape[1]):
                if i < len(predictions_dict[self.model_keys[j]]):
                    predictions_array[i, j] = predictions_dict[self.model_keys[j]][i]
                else:
                    predictions_array[i, j] = None

        return predictions_array
