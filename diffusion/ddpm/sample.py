import os
import sys
import torch
import argparse
import torchvision
from PIL import Image
from utils import get_diffusion_defaults, add_dict_to_argparser, build_diffusion, load_yaml, dict2namespace

def make_and_save_gif(frames_list, out_name='test.gif'):
    frame_one = frames_list[0]
    frame_one.save(out_name, format="GIF", append_images=frames_list, save_all=True, duration=100, loop=0)

def main():
    config_path = sys.argv[1]
    yaml_dict = load_yaml(config_path)
    args = dict2namespace(yaml_dict)
    os.makedirs(args.sampling.save_dir, exist_ok=True)

    print(args)
    diffusion = build_diffusion(argparse.Namespace(**vars(args.diffusion), **vars(args.data))).to(args.config.device)

    try:
        state_dict =  torch.load(args.sampling.model_path, map_location='cpu')
        diffusion.load_state_dict(state_dict)
        
        frames_list = []
        if args.diffusion.use_labels:
            images_per_class = args.sampling.num_images // 10
            for label in range(10):
                y = torch.ones(images_per_class, dtype=torch.long, device=args.config.device) * label
                samples_processes = diffusion.sample_diffusion_sequence(images_per_class, args.config.device, y=y)
                for process in samples_processes:
                    grid_sample = torchvision.utils.make_grid(process, nrow=4)
                    ndarr = grid_sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    img = Image.fromarray(ndarr)
                    frames_list.append(img)
                make_and_save_gif(frames_list, os.path.join(args.sampling.save_dir, 'sample_process.gif'))
                frames_list[-1].save(os.path.join(args.sampling.save_dir, "sample_results.png"))
        else:
            samples_processes = diffusion.sample_diffusion_sequence(args.sampling.num_samples, args.config.device, y=None)
            for process in samples_processes:
                grid_sample = torchvision.utils.make_grid(process, nrow=4)
                ndarr = grid_sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                img = Image.fromarray(ndarr)
                frames_list.append(img)
            make_and_save_gif(frames_list, 'sample_process.gif')
            frames_list[-1].save("sample_results.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")
                
if __name__ == '__main__':
    main()
    
from PIL import Image
reshaped_array = None
