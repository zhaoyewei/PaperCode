#!/usr/bin/env python
# coding: utf-8

# # 实现Resnet18

# In[19]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# In[2]:


class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(BasicBlock,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,
                      stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,
                     stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.shortcut =  nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,
                         stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out = self.seq(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


# In[3]:


class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResBlock,self).__init__()
        planes = 4
        self.seq = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,
                      stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel*planes,kernel_size=1,stride=1,
                     bias=False),
            nn.BatchNorm2d(outchannel*planes),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel * planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel*planes,kernel_size=1,
                               stride=stride,bias=False),
                nn.BatchNorm2d(outchannel*planes)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.seq(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


# In[4]:


# class ResNet18(nn.Module):
#     def __init__(self,block,num_classes=10):
#         super(ResNet18,self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
#         )
#         self.layer1 = self.make_layers(block,64,2,stride=1)
#         self.layer2 = self.make_layers(block,128,2,stride=2)
#         self.layer3 = self.make_layers(block,256,2,stride=2)
#         self.layer4 = self.make_layers(block,512,2,stride=2)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512,num_classes)
#     def make_layers(self,block,outchannel,num_blocks,stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel,outchannel,stride))
#             self.inchannel = outchannel
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x,1)
#         x = self.fc(x)
#         return x


# In[5]:


# class ResNet34(nn.Module):
#     def __init__(self,block,num_classes=10):
#         super(ResNet34,self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
#         )
#         self.layer1 = self.make_layers(block,64,3,stride=1)
#         self.layer2 = self.make_layers(block,128,4,stride=2)
#         self.layer3 = self.make_layers(block,256,6,stride=2)
#         self.layer4 = self.make_layers(block,512,3,stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512,num_classes)
#     def make_layers(self,block,outchannel,num_blocks,stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel,outchannel,stride))
#             self.inchannel = outchannel
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x,1)
#         x = self.fc(x)
#         return x


# In[6]:


# class ResNet50(nn.Module):
#     def __init__(self,block,num_classes=100):
#         super(ResNet50,self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3,self.inchannel,7,stride=2,padding=3,bias=False),
#             nn.BatchNorm2d(self.inchannel),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
#         )
#         self.layer1 = self.make_layers(block, 64,3,stride=1)
#         self.layer2 = self.make_layers(block,128,4,stride=2)
#         self.layer3 = self.make_layers(block,256,6,stride=2)
#         self.layer4 = self.make_layers(block,512,3,stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512 * 4,num_classes)
#     def make_layers(self,block,outchannel,num_blocks,stride=1):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel,outchannel,stride=stride))
#             self.inchannel = outchannel * 4
        
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.fc(x)
#         return x


# In[7]:


# net = ResNet50(ResBlock).to(device)

# #define loss funtion & optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# #train
# for epoch in range(pre_epoch, EPOCH):
#     print('\nEpoch: %d' % (epoch + 1))
#     net.train()
#     sum_loss = 0.0
#     correct = 0.0
#     total = 0.0
#     for i, data in enumerate(trainloader, 0):
#         #prepare dataset
#         length = len(trainloader)
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
        
#         #forward & backward
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         #print ac & loss in each batch
#         sum_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += predicted.eq(labels.data).cpu().sum()
#         print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
#               % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        
#     #get the ac with testdataset in each epoch
#     print('Waiting Test...')
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for data in testloader:
#             net.eval()
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum()
#         print('Test\'s ac is: %.3f%%' % (100 * correct / total))

# print('Train has finished, total epoch is %d' % EPOCH)


# In[8]:


# class ResNet101(nn.Module):
#     def __init__(self,block,num_classes=100):
#         super(ResNet101,self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3,self.inchannel,7,stride=2,padding=3,bias=False),
#             nn.BatchNorm2d(self.inchannel),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
#         )
#         self.layer1 = self.make_layers(block, 64,3,stride=1)
#         self.layer2 = self.make_layers(block,128,4,stride=2)
#         self.layer3 = self.make_layers(block,256,23,stride=2)
#         self.layer4 = self.make_layers(block,512,3,stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512 * 4,num_classes)
#     def make_layers(self,block,outchannel,num_blocks,stride=1):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel,outchannel,stride=stride))
#             self.inchannel = outchannel * 4
        
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.fc(x)
#         return x


# In[9]:


class ResNet(nn.Module):
    def __init__(self,block,num_layers,plane=4,num_classes=100):
        super(ResNet,self).__init__()
        self.inchannel = 64
        self.plane = plane
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,self.inchannel,7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self.make_layers(block, 64,num_layers[0],stride=1)
        self.layer2 = self.make_layers(block,128,num_layers[1],stride=2)
        self.layer3 = self.make_layers(block,256,num_layers[2],stride=2)
        self.layer4 = self.make_layers(block,512,num_layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.plane,num_classes)
    def make_layers(self,block,outchannel,num_blocks,stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel,outchannel,stride=stride))
            self.inchannel = outchannel * self.plane
        
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


# In[10]:


def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2],plane=1)
def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3],plane=1)
def ResNet50():
    return ResNet(ResBlock,[3,4,6,3],plane=4)
def ResNet101():
    return ResNet(ResBlock,[3,4,23,3],plane=4)
def ResNet152():
    return ResNet(ResBlock,[3,8,36,3],plane=4)


# In[17]:


# 超参数设置
EPOCH = 135   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 1024      #批处理尺寸(batch_size)
LR = 0.01        #学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[20]:


def train_all():
    models = [ResNet18(),ResNet34(),ResNet50(),ResNet101(),ResNet152()]
    model_names = ["R18","R34","R50","R101","R152"]
    for i in range(len(models)):
        print(model_names[i])
        with open("ResNet.txt","a") as f:
            f.write(model_names[i]+"\n")
        train(models[i])
    
    
def train(net):
    net = net.to(device)

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)
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
            with open("ResNet.txt","a") as f:
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
            with open("ResNet.txt","a") as f:
                f.write('Test\'s ac is: %.3f%%\n' % (100 * correct / total))

    print('Train has finished, total epoch is %d\n' % EPOCH)


# In[21]:


train_all()

