import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder


default_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587])
])

no_transforms = transforms.Compose([transforms.ToTensor(),])

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_classes(self):
        return self.dataset.classes
    

