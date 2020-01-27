from core.dataloader import *
dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv', split=0.80,
                     cols=['Adj Close', 'Volume'],
                     label_col='Adj Close', MinMax=False)

timesteps = 5
train_dt = dataset.get_train_data(timesteps, False, 2)
test_dt = dataset.get_test_data(timesteps, False, 2)
for i in range(10):
    print(train_dt[0][i])
    print(train_dt[1][i])