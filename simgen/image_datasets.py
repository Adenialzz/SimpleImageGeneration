import os.path as osp
from glob import glob
from PIL import Image
import random
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Dataset

def get_default_transform(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform


class MnistDataset(Dataset):
    def __init__(self, data_root, img_size=28):
        super().__init__()
        self._mnist = MNIST(data_root, train=True, download=True)
        self.img_size = img_size
        self.img_channels = 1
        self.transform = get_default_transform(self.img_size)

    def __len__(self):
        return len(self._mnist)
    
    def __getitem__(self, idx):
        img, label = self._mnist[idx]
        img_tensor = self.transform(img)
        return img_tensor

class CifarDataset(Dataset):
    def __init__(self, data_root, img_size=32):
        super().__init__()
        self._cifar = MNIST(data_root, train=True, download=True)
        self.img_size = img_size
        self.img_channels = 3
        self.transform = get_default_transform(self.img_size)

    def __len__(self):
        return len(self._cifar)
    
    def __getitem__(self, idx):
        img, label = self._mnist[idx]
        img_tensor = self.transform(img)
        return img_tensor

class CelebaDataset(Dataset):
    def __init__(self, data_root, img_size=256):
        super().__init__()
        
        self.img_size = img_size
        self.img_channels = 3
        self.all_image_paths = glob(f"{data_root}/**/*.jpg", recursive=True)
        random.shuffle(self.all_image_paths)
    
    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img_pil)
        return img_tensor



def get_dataset_class(dataset_name):
    assert dataset_name.lower() in ('mnist', 'cifar10', 'celeba'), f"dataset name = {dataset_name} is unknown, only mnist, cifar10, celeba supported"
    if dataset_name.lower() == 'mnist':
        return MnistDataset
    elif dataset_name.lower() == 'cifar10':
        return CifarDataset
    elif dataset_name.lower() == 'celeba':
        return CelebaDataset

if __name__ == '__main__':
    name = 'mnist'
    root = './data'
    dataset = get_dataset_class(name)(root, img_size=28)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4)
    for img in dataloader:
        print(img.shape)