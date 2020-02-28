import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
from core.hybrid_model import *
from hypopt_results import *
# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)

h = 7
dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/GSPC.csv', split=0.7,
                     cols=['log_ret'],
                     label_col='log_ret', MinMax=False)
timesteps = 29
train_dt = dataset.get_train_data(timesteps, False, 1)
test_dt = dataset.get_test_data(timesteps, False, 1)
print(train_dt[0:2])
# Check if the data is using right labels with num forward:
#print(test_dt[0][0], test_dt[1][0])
#print(test_dt[0][1], test_dt[1][1])
# print(test_dt[0][2], test_dt[1][2])
# print(test_dt[0][3], test_dt[1][3])
# print(test_dt[0][4], test_dt[1][4])
# print(test_dt[0][5], test_dt[1][5])
# print(test_dt[0][6], test_dt[1][6])

# Parameters
dataloader_params_train = {'batch_size': 1,
                           'shuffle': True,
                           'drop_last': True,
                           'num_workers': 0}
# Parameters
dataloader_params_test = {'batch_size': 1,
                          'shuffle': False,
                          'drop_last': True,
                          'num_workers': 0}
# Generators
training_set = Dataset(train_dt)
print(training_set)
training_generator = data.DataLoader(training_set, **dataloader_params_train)
test_set = Dataset(test_dt)
test_generator = data.DataLoader(test_set, **dataloader_params_test)
network_params = {'input_dim': 1,  # As many as there are of columns in data
                  'hidden_dim': h,
                  'batch_size': dataloader_params_train['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }
epochs = 150

folder_name = 'Training_best_one_on_GSPC'
new_folder = create_folder(path + '/results', folder_name)
Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_optimiser = torch.optim.Adam(Nice_model.parameters(), lr=0.0007255273517151122)
#Nice_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Nice_optimiser, epochs)
Nice_model_trained, loss_train_mtrx, loss_test_mtrx, error = train_model(Nice_model, Nice_loss, Nice_optimiser,
                                                                         None, epochs,
                                                                         training_generator,
                                                                         test_generator, timesteps,
                                                                         dataloader_params_train['batch_size'],
                                                                         new_folder, False)
for model in ['checkpoint', "last_model"]:
    path_to_checkpoint = new_folder + '/' + model + '.pth.tar'
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(path_to_checkpoint)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(path_to_checkpoint,
                                map_location=lambda storage,
                                                    loc: storage)
    Nice_model.load_state_dict(checkpoint['state_dict'])
    ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train = eval_model(Nice_model,
                                                                                                            Nice_loss,
                                                                                                            dataset,
                                                                                                            timesteps,
                                                                                                            1)
    y_training = train_dt[1]
    y_testing = test_dt[1]
    plot_and_save(ys, ys_testing, y_training, y_testing, loss_train_mtrx, loss_test_mtrx,
                  loss_vals_train,
                  loss_vals_test, False, model,
                  new_folder)

    #model_keys = ['1forward', '2forward', '3forward', '4forward']
    #config = {'1forward': {'hidden_dim': 4, 'num_layers': 1, 'timesteps': 15, 'state_dict': checkpoint['state_dict'],
    #                       'num_forward': 1},
    #          '2forward': {'hidden_dim': 4, 'num_layers': 1, 'timesteps': 10, 'state_dict': checkpoint['state_dict'],
    #                       'num_forward': 2},
    #          '3forward': {'hidden_dim': 4, 'num_layers': 1, 'timesteps': 19, 'state_dict': checkpoint['state_dict'],
    #                       'num_forward': 3},
    #          '4forward': {'hidden_dim': 4, 'num_layers': 1, 'timesteps': 21, 'state_dict': checkpoint['state_dict'],
    #                       'num_forward': 4},
    #          }
    #test_hybrid = hybrid_model(16, model_keys, config_multiple_models)
    #test_hybrid.run_predictions(dataset, 10, True)
    #sequences = test_hybrid.get_predictions()
    ## test_dt = dataset.get_test_data(timesteps, False, 1)
    #y_testing = test_dt[1]
    ## sequences = predict_seq_avg(Nice_model, test_dt[0], timesteps, 15)
    #plot_results_multiple(sequences, y_testing, 10, new_folder)
