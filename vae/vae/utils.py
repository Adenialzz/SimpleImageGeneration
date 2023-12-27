import torch
import os
import os.path as osp

def save_model(model_dir, epoch, model, optimizer):
    if not osp.isdir(model_dir):
        os.mkdir(model_dir)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, osp.join(model_dir, f"model_{epoch}.pth"))
