import argparse
import torchvision
import torch.nn.functional as F

import yaml
import argparse

from ddpm.unet import UNet
from ddpm.transform import RescaleChannels
from ddpm.diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)

def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def build_transform(img_size):

    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def get_diffusion_defaults():
    defaults = dict(
        img_channels=3,
        img_size=32,

        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,
        
        num_samples=64,
        n_rows = 8,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
        
        schedule_low=1e-4,
        schedule_high=0.02
    )

    return defaults


def build_diffusion(args) -> GaussianDiffusion:
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=args.img_channels,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 10,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (args.img_size, args.img_size), args.img_channels, None if not args.use_labels else 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion