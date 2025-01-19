# pytorch-神经网络搭建

> Neural Network

## nn.module基本使用

```python
def forward(self,x):
x = F.relu(self,conv1(x))#卷积+非线性处理
return F.relu(self.conv2(x))#两次后输出
```

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.model):
    def __init__(self):
        super().__init__()
        
    def forward(self,input):
        output = input + 1
        return output
    
model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
```

## 卷积（特征提取）

> 某尺寸的卷积核放入输入图像，输出卷积核内乘积的和(权重)

```python
import torch
import torch.nn.functional as F
#输入图像
input = torch.tensor([1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1])
#卷积核
kernal = torch.tensor([1,2,1],
                      [0,1,0],
                      [2,1,0])
#参数为batch_size,通道数，高，宽
input = torch.reshape(input,(1,1,5,5))
kernal = torch.reshape(kernal,(1,1,3,3))

output = F.conv2d(input,kernal,stride=1)
```

``torch.nn.Conv2d(in_channels,out_channels,kernal_size,stride=1,padding=0,``

``dilation=1,groups=1,bias=True,padding_mode='zeros')``

``out_channels``=卷积核数

```python
import torch
import torchvision

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernal_size=3,stride=1,padding=0)
        
    def forward(self,x):
        x = self.conv1(x)
        return x
    
model = Model()
step = 0
writer = SummaryWriter("../logs")
for data in dataloader:
    imgs,targets = data
    output = model(imgs)
    print(imgs.shape)#(64,3,32,32)
    print(output.shape)#(64,6,30,30)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))#转回3通道
    writer.add_images("output",output,step)

writer.close()
```

## 最大池化

> 池化核内取最大值，压缩原图像并保留特征

```python
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

input = torch.tensor([1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1])
input = torch.reshape(input,(-1,1,5,5))

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.maxpool1 = MaxPool2d(kernal_size=3,ceil_mode=True)
    
    def forward(self,input):
        output = self.maxpool1(input)
        return output
model = Model()
output = model(input)
print(output)
step = 0
writer = SummaryWriter("../logs")
for data in dataloader:
    imgs,targets = data
    output = model(imgs)
    writer.add_images("output",output,step)
    step = step + 1
    
writer.close()
```

## 非线性激活

> 为神经网络引入了非线性因素，使得神经网络能够学习和模拟复杂的非线性关系

```python
import torch
import torch.nn as nn
import torchvision
from torch.nn import ReLU
from torch.utils.data import DataLoader

input = torch.tensor([1,-0.5],
                     [-1,3])
input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.relu1 = ReLU()
    
    def forward(self,input):
        output = self.relu1(input)
        return output

model = Model()
output = model(input)
print(output)
step = 0
writer = SummaryWriter("../logs")
for data in dataloader:
    imgs,targets = data
    output = model(imgs)
    writer.add_images("output",output,step)
    step = step + 1
    
writer.close()
```

## 线性层

```python
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = Linear(196608,10)
        
    def forward(self,input):
        output = self.linear1(input)
        return output
    
model = Model()

for data in dataloader:
    imgs,targets = data
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
```

## 实战&Sequential使用

```python
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__  
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
        
    def forward(self,x):
        x = self.model1(x)
        return x

model = Model()
input = torch.ones((64,3,32,32))
output = model(input)
print(outpu.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(model,input)
writer.close()
```

