#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
# torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# In[ ]:


def conv_bn(inp,oup,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(inp,oup,kernel_size=kernel_size,stride=stride,padding=1,bias=False)),
        ("norm",nn.BatchNorm2d(oup)),
        ("relu",nn.ReLU(inplace=True))
    ]))


# In[ ]:


class OSA_Module(nn.Module):
    def __init__(self,inp,oup,ouc,stride=1,layer_per_block=5,use_maxpool=True):
        '''
        inp：当前OSA Module的输入通道
        oup：当前OSA Module在concat之前的通道数
        ouc：当前OSA Module在concat之后的通道数
        stride：当前OSA Module用到的步长
        layer_per_block：每个OSA Module有几层
        '''
        super(OSA_Module,self).__init__()
#         self.maxpool = nn.ModuleList()
#         if use_maxpool:
#             self.maxpool.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.blocks = nn.ModuleList()
        for i in range(layer_per_block):
            if i == 0:
                '''
                输入为上一个OSA Module的输出，输入通道为inp
                '''
                self.blocks.append(conv_bn(inp,oup))
            else:
                '''
                输入为本OSA Module上一层的输出，输入通道为oup
                '''
                self.blocks.append(conv_bn(oup,oup))
        '''
        从后往前，之后相加，为inp加上layer_per_block个oup
        '''
        inc = inp + oup * layer_per_block
        self.concat = nn.Sequential(
            nn.Conv2d(inc,ouc,kernel_size=1,
                      padding=0,bias=False),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
        )
        
    def forward(self,x):
#         for maxpool in self.maxpool:
#             x = maxpool(x)
        outputs = [x]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        x = torch.cat(outputs,dim=1)
        x = self.concat(x)
        return x


# In[ ]:


class VoVNet(nn.Module):
    def __init__(self,oups,oucs,num_layers,num_classes=10,layer_per_block=5):
        '''
        oups：每一个stage在concat之前的输出通道
        oucs：每一个stage在concat之后的输出通道数
        num_layers：每个stage有几个OSA Module
        '''
        super(VoVNet,self).__init__()
        self.inp = 128
        block = OSA_Module
        '''
            第一层 stem stage：
            从3->64，64->64，最后输出128个通道，同时输出大小减半
        '''
        self.Stem_Stage_1 = nn.Sequential(
                         OrderedDict([
                             ("stem1",conv_bn(3,64,stride=2)),
                             ("stem2",conv_bn(64,64,stride=1)),
                             ("stem3",conv_bn(64,128,stride=1))
                         ]))
        '''
        stage 2:表示输入是上一个stage的输出，故输入为self.inp
        '''
        self.OSA_Stage_2 = self.make_layers(block,self.inp,oups[0],oucs[0],num_layers[0],2,False)
        '''
        否则就是本层stage的上一个OSA Module的输出
        '''
        self.OSA_Stage_3 = self.make_layers(block,oucs[0],oups[1],oucs[1],num_layers[1],3,True)
        self.OSA_Stage_4 = self.make_layers(block,oucs[1],oups[2],oucs[2],num_layers[2],4,True)
        self.OSA_Stage_5 = self.make_layers(block,oucs[2],oups[3],oucs[3],num_layers[3],5,True)
        
#         '''
#         对每个stage建立对应数目的OSA Module
#         '''
#         for i in range(len(num_layers)):
#             if i == 0:
#                 '''
#                 i==
#                 '''
#                 self.layers.append(self.make_layers(block,self.inp,oups[i],oucs[i],num_layers[i],i+2))
#                 self.inp = oucs[-1]
#             else:
#                 '''
#                 否则就是本层stage的上一个OSA Module的输出
#                 '''
#                 self.layers.append(self.make_layers(block,oups[i-1],oups[i],oucs[i],num_layers[i],i+2))
            
        self.fc = nn.Linear(oucs[3],num_classes)
    def make_layers(self,block,inp,oup,ouc,num_layers,dep,use_maxpool=True,layers_per_block=5):
        '''
        block：模块，这里是OSA
        inp：当前OSA Module的输入通道数
        oup：当前OSA Module在concat之前的通道数
        ouc：当前OSA Module在concat之后的通道数
        num_layers：本层有多少个OSA Module
        dep：第几个stage，用来写名字用
        '''
        layers = []
        for i in range(num_layers):
            layers.append(block(inp,oup,ouc,use_maxpool=use_maxpool))
            inp = ouc
        layers.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.Stem_Stage_1(x)
        x = self.OSA_Stage_2(x)
        x = self.OSA_Stage_3(x)
        x = self.OSA_Stage_4(x)
        x = self.OSA_Stage_5(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


# In[ ]:


def VoVNet27():
    return VoVNet([64,80,96,112],[128,256,384,512],[1,1,1,1])


# In[ ]:


def VoVNet39():
    return VoVNet([128,160,192,224],[256,512,768,1024],[1,1,2,2])


# In[ ]:


def VoVNet57():
    return VoVNet([128,160,192,224],[256,512,768,1024],[1,1,4,3])


# In[ ]:


# 超参数设置
EPOCH = 100   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.01        #学习率


# In[ ]:


# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(32),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:


filename = 'VoVNet.txt'
def train_all():
    models = [VoVNet57()]
    model_names = ["V27","V39","V57"]
    for i in range(len(models)):
        print(model_names[i])
        with open(filename,"a") as f:
            f.write(model_names[i]+"\n")
        train(models[i])
    
    
def train(net):
    net = net.to(device)
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
    
# train_all()
