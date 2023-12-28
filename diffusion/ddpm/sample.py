import os
import torch
import argparse
import torchvision
from PIL import Image
from diffusion.ddpm.utils import diffusion_defaults, add_dict_to_argparser, get_diffusion_from_args

def make_and_save_gif(frames_list, out_name='test.gif'):
    frame_one = frames_list[0]
    frame_one.save(out_name, format="GIF", append_images=frames_list, save_all=True, duration=100, loop=0)

def create_argparser():
    defaults = dict(num_images=100, device='cuda:1')
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--viz_process", action='store_true', default=False)
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        diffusion = get_diffusion_from_args(args).to(args.device)
        state_dict =  torch.load(args.model_path, map_location='cpu')
        diffusion.load_state_dict(state_dict)
        
        frames_list = []
        if args.use_labels:
            images_per_class = args.num_images // 10
            for label in range(10):
                y = torch.ones(images_per_class, dtype=torch.long, device=args.device) * label
                samples_processes = diffusion.sample_diffusion_sequence(images_per_class, args.device, y=y)
                for process in samples_processes:
                    grid_sample = torchvision.utils.make_grid(process, nrow=4)
                    ndarr = grid_sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    img = Image.fromarray(ndarr)
                    frames_list.append(img)
                make_and_save_gif(frames_list, 'sample_process.gif')
                frames_list[-1].save("sample_results.png")
        else:
            samples_processes = diffusion.sample_diffusion_sequence(args.num_images, args.device, y=None)
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
