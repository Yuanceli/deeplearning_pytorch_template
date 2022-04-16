import pickle

import torch
from torch import nn


def save_pickle(info, filename):
    with open(filename, "wb") as f:
        pickle.dump(info, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        infos = pickle.load(f)
    return infos


def save_model(model, optimizer, epoch, batch, loss, absolute_path):
    if isinstance(model, nn.DataParallel):
        my_model = model.module
    else:
        my_model = model
    torch.save({'epoch': epoch,
                'model_state_dict': my_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, absolute_path)
