# 网络模型

## 现有的使用和修改

```cmd
pip install 模型库名
```

```python
import torchvision
from torch import nn

train_data = torchvision.datasets.ImageNet("../data_image_net",split='train',download=True,
                                          transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)#训练好的需要下载

train_data = torchvision.datasets.CIFAR10('../data',train=True,download=True,
                                         transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear',nn.linear(1000,10))#分类神经网络增加结构
vgg16_true.classifier[6] = nn.Linear(4096,10)#分类神经网络修改结构
```

## 模型的保存与读取

```python
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

#save1:structure+parameter
torch.save(vgg16,"vgg16_method1.pth")

#save2：parameter
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
```

```python
import torch

#save1
model1 = torch.load("vgg16_method1.pth")

#save2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method1.pth"))
print(vgg16)
```

