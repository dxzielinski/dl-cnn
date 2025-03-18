import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder


default_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    # hue shifts color around the RGB color wheel. +-0.05 * 365deg ~= +-18deg
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(1.2, 1.8)),  # cutout
    # 32*32 = 1024; 0.005 * 1024 ~= 5; 3x2 has ratio 1.5, so ratio can be 1.2-1.8
    transforms.Normalize(mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587])
])

no_transforms = transforms.Compose([transforms.ToTensor(),])

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, batch_size=32):
        self.dataset = ImageFolder(root, transform=transform)
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_classes(self):
        return self.dataset.classes
    
    def get_dataloader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=True, num_workers=4)
