#!/usr/bin/env python
# coding: utf-8

# In[130]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# In[131]:


def conv_bn(inp,oup,stride):
    return nn.Sequential(
        nn.Conv2d(inp,oup,kernel_size=3,stride=stride,padding=1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
class InvertedResidual(nn.Module):
    def __init__(self,inp,oup,stride,expand_ratio):
        '''
        t：扩展因子
        c：输出特征矩阵深度
        n：bottleneck重复次数
        s：步距离
        '''
        super(InvertedResidual,self).__init__()
        hidden_dim = inp*expand_ratio
        self.stride = stride
        self.use_res_connect = (self.stride==1 and inp == oup)  
        # expannd_ratio == 1代表第一个bottlenneck不需要其一个1x1卷积，因此不用要
        if expand_ratio == 1:
            self.seq = nn.Sequential(
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=stride,
                          padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,oup,kernel_size=1,stride=1,
                          padding=0,bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            #倒残差的情况，大部分都是这个情况
            self.seq = nn.Sequential(
                nn.Conv2d(inp,hidden_dim,kernel_size=1,
                          stride=1,padding=0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,
                         stride=stride,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,oup,kernel_size=1,
                         stride=1,padding=0,bias=False),
                nn.BatchNorm2d(oup)
            )
        self.relu = nn.ReLU6(inplace=True)
    def forward(self,x):
        if self.use_res_connect:
            return x + self.seq(x)
        return self.seq(x)


# In[132]:


class MobileNetV2(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNetV2,self).__init__()
        block = InvertedResidual
        inp = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.features = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(3,inp,kernel_size=3,stride=2,
                          padding=1,bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True)
            )])
        self.conv = nn.Sequential(
            nn.Conv2d(320,1280,kernel_size=1,stride=1,
                                padding=0,bias=False),
            nn.AvgPool2d(7,stride=1),
            nn.Conv2d(1280,num_classes,kernel_size=1,
                    stride=1,padding=0,bias=False)
        )
        layers = []
        for t,c,n,s in interverted_residual_setting:
            for i in range(n):
                if i == 0:
                    self.features.append(block(inp,c,s,t))
                else:
                    self.features.append(block(inp,c,1,t))
                inp = c
        self.fc = nn.Linear(num_classes,num_classes,bias=False)
    def forward(self,x):
        for m in self.features:
            x = m(x)
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


# In[133]:


# 超参数设置
EPOCH = 135   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.01        #学习率


# In[134]:


# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[136]:


# def train_all():
#     models = [ResNet18(),ResNet34(),ResNet50(),ResNet101(),ResNet152()]
#     model_names = ["R18","R34","R50","R101","R152"]
#     for i in range(len(models)):
#         print(model_names[i])
#         with open("ResNet.txt","a") as f:
#             f.write(model_names[i]+"\n")
#         train(models[i])
    
    
# def train(net):
net = MobileNetV2().to(device)
filename = 'MObileNetV2.txt'
#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#train
for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        #prepare dataset
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        #forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        with open(filename,"a") as f:
            f.write('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% \n' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

    #get the ac with testdataset in each epoch

    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test\'s ac is: %.3f%%' % (100 * correct / total))
        with open(filename,"a") as f:
            f.write('Test\'s ac is: %.3f%%\n' % (100 * correct / total))

print('Train has finished, total epoch is %d\n' % EPOCH)


# In[ ]:
