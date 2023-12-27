import torch
import datetime
import argparse
import torchvision  
from tqdm import tqdm
from ddpm.script_utils import diffusion_defaults, add_dict_to_argparser, get_diffusion_from_args, get_transform, cycle

def create_parser():
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        log_to='tensorboard',
        log_rate=500,
        checkpoint_rate=500,
        log_dir="./logs/ddpm_logs",
        project_name="ddpm",
        run_name=datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M"),

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device='cuda',
    )
    
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_parser().parse_args()
    device = args.device
    
    try:
        diffusion = get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        
        if args.model_checkpoint is not None:
            state_dict = torch.load(args.model_checkpoint, map_location='cpu')
            diffusion.load_state_dict(state_dict)
        if args.optim_checkpoint is not None:
            state_dict = torch.load(args.optim_checkpoint, map_location='cpu')
            diffusion.load_state_dict(state_dict)
        if args.log_to == 'wandb':
            import wandb
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")
            
            run = wandb.init(project=args.project_name, entity='treaptofun', config=vars(args), name=args.run_name)
            wandb.watch(diffusion)
        elif args.log_to == 'tensorboard':
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(args.log_dir)
        else:
            print('Warning: NO logs')
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transform())
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transform())
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=2)
        train_loader = cycle(train_loader)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=2)
        
        train_loss = 0.0
        for iteration in tqdm(range(1, args.iterations + 1)):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(args.device)
            y = y.to(args.device)
            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x, None)
            
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            diffusion.update_ema()
            
            if iteration % args.log_rate == 0:
                test_loss = 0.0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(args.device)
                        y = y.to(args.device)
                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x, None)
                        
                        test_loss += loss.item()
                
                if args.use_labels:
                    samples = diffusion.sample(16, args.device, y=torch.arange(10, device=args.device))
                else:
                    samples = diffusion.sample(16, args.device, y=None)
                                    
                samples = ((samples + 1) / 2).clip(0, 1)
                test_loss /= len(test_loader)
                train_loss /= args.log_rate
                
                if args.log_to == 'wandb':
                    samples = samples.permute(0, 2, 3, 1).numpy()
                    wandb.log( { 'test_loss': test_loss, "train_loss": train_loss, "samples": [wandb.Image(sample) for sample in samples] })
                elif args.log_to == 'tensorboard':
                    writer.add_scalar('test loss', test_loss, iteration)
                    writer.add_scalar('train loss', train_loss, iteration)
                    writer.add_images('samples', samples, iteration)

                train_loss = 0.0
                
                            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                
            if args.log_to == 'wandb':
                run.finish()
    except KeyboardInterrupt:
        if args.log_to == 'wandb':
            run.finish()
        print("Keyboard interrupt, run finished early")

if __name__ == '__main__':
    main()
