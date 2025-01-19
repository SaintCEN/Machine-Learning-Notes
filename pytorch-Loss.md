# Loss

## 损失函数

> 计算输出和目标之间的差距，为更新输出提供一定的依据

$$
交叉熵公式：loss(x,class)=-x[class]+log(\sum_jexp(x[j]))
$$

```python
import torch
from torch import nn

inputs = torch.tensor([1,2,3])
targets = torch.tensor([1,2,5])

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.tensor(targets,(1,1,1,3))

loss = L1Loss()
result1 = loss(inputs,targets)
loss_MSE = MSEloss()
result2 = MSEloss(inputs,targets)
print(result1)
print(result2)

#交叉熵-分类问题
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.crossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)
```

```python
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

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
loss_cross = nn.crossEntropyLoss()

for data in dataloader:
    imgs,targets = data
    outputs = model(imgs)
    result_loss = loss_cross(outputs,targets)
    result_loss.backward()#反向传播
    print(result_loss)
```

## 优化器

> 梯度下降法

```python
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential,optim
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transfrom.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset,batch_size=64)

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
loss_cross = nn.crossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=0.01)#随机梯度下降，lr为学习率

for epoch in range(20):
    running_loss = 0.0
for data in dataloader:
    imgs,targets = data
    outputs = model(imgs)
    result_loss = loss_cross(outputs,targets)
    optim.zero_grad()#重置
    result_loss.backward()
    optim.step()#参数调优
    running_loss += result_loss
print(running_loss)
```

