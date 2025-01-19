import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter

img_path = '../dataset/train/dog/dog.jpg'
img = Image.open(img_path)
image = img.convert('RGB')

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transforms(image)

# print(image.shape)

class CIF(nn.Module):
    def __init__(self):
        super(CIF, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

writer = SummaryWriter('../logs')

model = torch.load('../code/cif_9.pth',map_location=torch.device('cpu'))# GPU训练的模型 cpu时需要映射过来
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
writer.add_graph(CIF(), image)
writer.add_image('CIF', image, dataformats='NCHW')

writer.close()