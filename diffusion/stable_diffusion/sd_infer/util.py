from models import CLIP, Encoder, Decoder, Diffusion
import warnings
import torch
import numpy as np
import os


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def get_file_path(filename, url=None):
    module_location = os.path.dirname(os.path.abspath(__file__))
    parent_location = os.path.dirname(module_location)
    file_location = os.path.join(parent_location, "data", filename)
    return file_location

def move_channel(image, to):
    if to == "first":
        return image.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    elif to == "last":
        return image.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    else:
        raise ValueError("to must be one of the following: first, last")

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x



def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace('_proj_weight', '_proj.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace('_proj_bias', '_proj.bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(("Given checkpoint data were modified dynamically by make_compatible"
                       " function on model_loader.py. Maybe this happened because you're"
                       " running newer codes with older checkpoint files. This behavior"
                       " (modify old checkpoints and notify rather than throw an error)"
                       " will be removed soon, so please download latest checkpoints file."))

    return state_dict

def load_clip(device):
    state_dict = torch.load(get_file_path('ckpt/clip.pt'))
    state_dict = make_compatible(state_dict)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict)
    return clip

def load_encoder(device):
    state_dict = torch.load(get_file_path('ckpt/encoder.pt'))
    state_dict = make_compatible(state_dict)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict)
    return encoder

def load_decoder(device):
    state_dict = torch.load(get_file_path('ckpt/decoder.pt'))
    state_dict = make_compatible(state_dict)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict)
    return decoder

def load_diffusion(device):
    state_dict = torch.load(get_file_path('ckpt/diffusion.pt'))
    state_dict = make_compatible(state_dict)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict)
    return diffusion

def preload_models(device):
    return {
        'clip': load_clip(device),
        'encoder': load_encoder(device),
        'decoder': load_decoder(device),
        'diffusion': load_diffusion(device),
    }