import os
import torch
import argparse
import torchvision
from ddpm.script_utils import diffusion_defaults, add_dict_to_argparser, get_diffusion_from_args

def create_argparser():
    defaults = dict(num_images=100, device='cuda:1')
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        diffusion = get_diffusion_from_args(args).to(args.device)
        state_dict =  torch.load(args.model_path, map_location='cpu')
        diffusion.load_state_dict(state_dict)
        
        if args.use_labels:
            images_per_class = args.num_images // 10
            for label in range(10):
                y = torch.ones(images_per_class, dtype=torch.long, device=args.device) * label
                samples = diffusion.sample(images_per_class, args.device, y=y)
                
                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:
            samples = diffusion.sample(args.num_images, args.device)

            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")
                
if __name__ == '__main__':
    main()
