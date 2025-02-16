import torch
import torchvision
import torchvision.transforms as transfroms

transfrom = transfroms.Compose(transfroms.ToTensor, transfroms.Normalize((0.5,), (0.5,)))

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transfrom)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transfrom)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)