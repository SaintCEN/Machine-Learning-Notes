# pytorch-DataProcess

> An Machine Learning Framework in Python

## Load Data

``torch.utils.data.Dataset``  ``torch.utils.data.DataLoader``

* Dataset：存储数据样本和期望值
* Dataloader：分批处理

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir#根目录
        self.label = label_dir#标签目录
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)#列表
    
    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,label_dir,img_name)
        img = Image.open(img.item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "dataset/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir,ants_label_dir)
```

```python
from torch.utils.data import DataLoader
import torchvision

#准备数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train = False,
                                         transform = torchvision.transforms.ToTensor())
#batch_size:堆中数据量；shuffle：true为随机，false为顺序；num_workers:子进程；drop_last:是否丢弃最后一个批次（不足4）
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=8,drop_last=False)

#第一张图以及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0;
for epoch in range(2):
for data in test_loader:
    imgs,targets = data
    print(image.shape)#会返回四个值
    print(target)
    writer.add_images("Epoch:{}".format(epoch),imgs,step)
    step = step + 1
    
writer.close()
```

## tensorboard

``add_image``:显示数据 ``add_schalar``:绘图

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")#写入日志

image_path = "data/train.123.jpg"#相对路径
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)#转为numpy型
writer.add_image("test",img_array,1,dataformats='HWC')#标题,数据，步长，格式

for i in range(100):
    writer.add_schalar("y=2x",2*i,i)#tag,y轴，x轴

writer.close()
```

```cmd
tensorboard --logdirs=logs --port=6007
```

## transforms

``ToTensor`` ``Normalize`` ``Resize``

```python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "data/train.123.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

#ToTensor转为张量
tensor_trans = transforms.ToTensor()#tensor类型
tensor_img = tensor_trans(img)
writer.add_image("tensor_img",tensor_img)
print(tensor_img)

#Normalize 归一化
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
writer.add_img("img_norm",img_norm)

#Resize改变大小
trans_resize = transform.Resize(512,512)
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)

trans_resize_2 = transform.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_2",img_resize_2,1)

#RandomCrop裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random",img_crop,1)
    
writer.close()
```

## torchvision

```python
import torchvision
#下载数据集
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.Totensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,
                                         transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,
                                        transform=dataset_transform,download=True)
print(test_set[0])
print(test_set.classes)

img,target = test_set[0]
print(img)
print(target)
ptint(test_set.classes[target])
img.show()

writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()
```

