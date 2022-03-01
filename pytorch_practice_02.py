import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

transf = tr.Compose([tr.Resize(16),tr.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='\data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root='\data', train=False, download=True, transform=transf)

print(trainset[0][0].size())

trainloader = DataLoader(trainset, batch_size = 50, shuffle = True)
testloader = DataLoader(testset, batch_size = 50, shuffle = True)

print(len(trainloader))

images, labels = iter(trainloader).next()
print(images.size())

oneshot = images[1].permute(1,2,0).numpy()

images[1].size()
images[1].permute(1,2,0).size()

# image[1])
# oneshot.size()

plt.figure(figsize=(2,2))
plt.imshow(oneshot)
plt.axis("off")
plt.show()