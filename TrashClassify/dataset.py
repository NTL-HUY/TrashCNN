from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = ImageFolder(
    root=r"D:\Projects\Workspace\Coding\Dataset\TrashType_Image_Dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print(dataset.classes)       # ['cat', 'dog']
print(dataset.class_to_idx)  # {'cat':0, 'dog':1}