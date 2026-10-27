from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


data_transformation=transforms.Compose([transforms.ToTensor])

train_dataset=datasets.MNIST(root='./data', train=True, transform=data_transformation, download=True)
validation_dataset=datasets.MNIST(root='./data', train=False, transform=data_transformation, download=True)

train_loader=DataLoader(train_dataset, batch_sampler=64, shuffle=True)
val_loader=DataLoader(validation_dataset,batch_size=64,shuffle=False)



