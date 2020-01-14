from torch.backends import cudnn
import torch
def pred_sequence(model,test_set,timesteps,start,num_preds):
    """'
    This function takes the model weights and predicts a sequence.
    Ie. It takes the #timesteps before start as created in test_gen and predicts
    a sequence of num_preds
    """

    batch, labels = test_set[start]
    batch = batch.view(timesteps, 1, -1)
    labels = torch.from_numpy(labels).type(torch.Tensor)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    # Transfer to GPU
    batch, labels = batch.to(device), labels.to(device)

    pred_sequence = []

    for i in range(num_preds):

        pred = model(batch)
        pred_sequence.append(pred)
        #print(batch)
    #print(pred_sequence)