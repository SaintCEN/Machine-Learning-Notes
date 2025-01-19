import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练设备,用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10('../dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10('../dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size)
print(test_data_size)

# 打包数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型  这块也可以单独放一个文件引入
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


cif = CIF()
cif.to(device)
# cif = cif.cuda()  #网络模型转移到cuda上
# 良好的写法：
# if torch.cuda.is_available():
#     cif = cif.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# loss_fn = loss_fn.cuda()  #损失函数转移到cuda

# 优化器
learning_rate = 1e-2  # 1x(10)^(-2)
optimizer = torch.optim.SGD(cif.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('../logs')

# 计算开始时间
start_time = time.time()

for i in range(epoch):
    print('-----------第{}轮训练开始------------'.format(i + 1))

    # 训练步骤开始
    cif.train()  # 和cif.eval()只对部分网络有用（dropout等）
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # imgs = imgs.cuda()  #图像数据加载到cuda
        # targets = targets.cuda()  #标签加载到cuda
        output = cif(imgs)
        loss = loss_fn(output, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  # 每100次打印一次
            end_time = time.time()
            print('训练时长:{}'.format(end_time - start_time))  # 计算总时长
            print('训练次数：{}, loss: {}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    cif.eval()
    total_test_loss = 0  # 损失指标
    total_accuracy = 0  # 准确率指标
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # imgs = imgs.cuda()  # 图像数据加载到cuda
            # targets = targets.cuda()  # 标签加载到cuda
            outputs = cif(imgs)
            loss = loss_fn(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()  # 1为横着方向,返回每一行最大的索引，True则返回1
            total_test_loss = total_test_loss + loss.item()
            total_accuracy = total_accuracy + accuracy

    print('整体测试集上的loss：{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(cif, 'cif_{}.pth'.format(i))
    print('模型已保存')

writer.close()