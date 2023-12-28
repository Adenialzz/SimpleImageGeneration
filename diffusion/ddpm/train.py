import os
import sys
import torch
import datetime
import torchvision  
from tqdm import tqdm
import argparse
from utils import build_diffusion, build_transform, cycle, load_yaml, dict2namespace

def parse_args(config_path):
    config = load_yaml(config_path)
    args = dict2namespace(config)
    args.config.run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    return args


def main():
    config_path = sys.argv[1]
    args = parse_args(config_path)
    os.makedirs(os.path.join(args.config.log_dir, args.config.run_name), exist_ok=True)
    
    try:
        diffusion = build_diffusion(argparse.Namespace(**vars(args.diffusion), **vars(args.data))).to(args.config.device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.training.lr)
        
        if args.config.model_checkpoint is not None:
            state_dict = torch.load(args.config.model_checkpoint, map_location='cpu')
            diffusion.load_state_dict(state_dict)
        if args.config.optim_checkpoint is not None:
            state_dict = torch.load(args.config.optim_checkpoint, map_location='cpu')
            diffusion.load_state_dict(state_dict)
        if args.config.log_to == 'wandb':
            import wandb
            if args.config.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")
            
            run = wandb.init(project=args.config.project_name, entity='treaptofun', config=vars(args), name=args.config.run_name)
            wandb.watch(diffusion)
        elif args.config.log_to == 'tensorboard':
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(args.config.log_dir)
        else:
            print('Warning: NO logs')
        
        if args.data.dataset.lower() == 'cifar10':
            assert args.data.img_size == 32, f"img size must be 32 when training on CIFAR10, not {args.data.img_size}"
            train_dataset = torchvision.datasets.CIFAR10(root=args.data.data_root, train=True, download=True, transform=build_transform(img_size=32))
            test_dataset = torchvision.datasets.CIFAR10(root=args.data.data_root, train=False, download=True, transform=build_transform(img_size=32))
        elif args.data.dataset.lower() == 'celeba':
            # celeba_root = '/home/jeeves/JJ_Projects/data/celeba_hq_256'
            train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data.data_root, 'train'), transform=build_transform(img_size=args.data.img_size))
            test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data.data_root, 'val'), transform=build_transform(img_size=args.data.img_size))
        else:
            raise ValueError(f"unspport dataset: {args.data.dataset}")
            
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.training.batch_size, drop_last=True, num_workers=2)
        train_loader = cycle(train_loader)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.training.batch_size, drop_last=True, num_workers=2)
        
        train_loss = 0.0
        for iteration in tqdm(range(1, args.training.iterations + 1)):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(args.config.device)
            y = y.to(args.config.device)
            loss = diffusion(x, y=y if args.diffusion.use_labels else None)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            diffusion.update_ema()
            
            if iteration % args.config.eval_freq == 0:
                test_loss = 0.0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(args.config.device)
                        y = y.to(args.config.device)
                        loss = diffusion(x, y=y if args.diffusion.use_labels else None)
                        test_loss += loss.item()
                
                samples = diffusion.sample(args.sampling.num_samples, args.config.device, y=torch.arange(10, device=args.config.device) if args.diffusion.use_labels else None)
                samples = ((samples + 1) / 2).clip(0, 1)
                test_loss /= len(test_loader)
                train_loss /= args.log_rate
                
                if args.config.log_to == 'wandb':
                    samples = samples.permute(0, 2, 3, 1).numpy()
                    wandb.log( { 'test_loss': test_loss, "train_loss": train_loss, "samples": [wandb.Image(sample) for sample in samples] })
                elif args.config.log_to == 'tensorboard':
                    writer.add_scalar('test loss', test_loss, iteration)
                    writer.add_scalar('train loss', train_loss, iteration)
                    writer.add_images('samples', samples, iteration)

                train_loss = 0.0
                
                            
            if iteration % args.config.checkpoint_freq == 0:
                model_filename = f"{args.config.log_dir}/{args.config.project_name}-{args.config.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.config.log_dir}/{args.config.project_name}-{args.config.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                
            if args.config.log_to == 'wandb':
                run.finish()
    except KeyboardInterrupt:
        if args.config.log_to == 'wandb':
            run.finish()
        print("Keyboard interrupt, run finished early")

if __name__ == '__main__':
    main()
