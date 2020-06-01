# CUDA_VISIBLE_DEVICES="0"
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
import struct
import torch.utils.data as Data 
import matplotlib.pyplot as plt



def min_max_normalization(features_matrix):#归一化
    max_ = []
    min_ = []
    for value in features_matrix[0]:  
        max_.append(value)
        min_.append(value)
    
    for j in range(len(features_matrix[0])):#求每一列的最大和最小值
        for i in range(len(features_matrix)):
            if max_[j] < features_matrix[i][j]:
                max_[j] = features_matrix[i][j]
            if min_[j] > features_matrix[i][j]:
                min_[j] = features_matrix[i][j]
    for i in range(len(features_matrix)):   #行
        for j in range(len(features_matrix[0])):#列
            if max_[j] == min_[j]:
                pass
            else:
                features_matrix[i][j] = (features_matrix[i][j]-min_[j]) / (max_[j]-min_[j])#归一化
                
    return np.array(features_matrix)

def getTrainData():
    triangle_signal = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\feature1.npy')#532*2*10*7*7
    triangle_GT = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\label1.npy')#532*28*28
    square_signal = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\feature2.npy')#532*2*10*7*7
    square_GT = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\label2.npy')#532*28*28
    
    train_signal = np.concatenate((triangle_signal,square_signal),axis=0)#1064*2
    train_GT = np.concatenate((triangle_GT,square_GT),axis=0)#1064*28
    return train_signal, train_GT

def getTestData():
    triangle_signal = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\feature1.npy')
    print(triangle_signal.shape)#行列数532*2
    triangle_GT = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\label1.npy')
    square_signal = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\feature2.npy')
    square_GT = np.load(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\label2.npy')
    
    test_signal = np.concatenate((triangle_signal,square_signal),axis=0)#1064*2
    test_GT = np.concatenate((triangle_GT,square_GT),axis=0)#1056*28
    return test_signal, test_GT

def loadData(batch_size):
    train_signal, train_GT = getTrainData()#signal:1064*2;GT:1064*28
    train_signal = torch.from_numpy(train_signal).float()#numpy数据转换成torch，为了去掉中括号
    train_GT = train_GT/255#归一化
    train_GT = torch.from_numpy(train_GT).float()#numpy数据转换成torch，为了去掉中括号
    train_signal1 = train_signal[:,:,:5]
    train_signal2 = train_signal[:,:,5:10]
    #train_signal = torch.cat([train_signal1,train_signal2],dim=0)
    #train_GT = torch.cat([train_GT,train_GT],dim=0)
    print(train_signal.size())#总共元素个数
    print(train_GT.size())
    
    
    test_signal, test_GT = getTestData()#signal:1064*2;GT:1064*28
    print(test_signal.shape)
    test_signal = torch.from_numpy(test_signal).float()#numpy数据转换成torch，为了去掉中括号
    #test_signal = test_signal[:,:,:5]
    test_GT = torch.from_numpy(test_GT).float()#numpy数据转换成torch，为了去掉中括号
    test_GT = test_GT/255
    
    
    torch_train = Data.TensorDataset(train_signal,train_GT)
    #torch_test = Data.TensorDataset(test_signal,test_GT)
    
    train_loader = Data.DataLoader(dataset = torch_train,
                             batch_size = batch_size,
                             shuffle=True)
    '''
    test_loader = Data.DataLoader(dataset=torch_test,
                                  batch_size = batch_size,
                                  shuffle=False)  
    '''
    return train_loader,test_signal,test_GT


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)#in_channal,out_channel,kernel_size,stride=1步长,padding=0,dilation=1kernal点间距,groups=1卷积核个数,bias=true(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)#对每一个batch的channel求均值方差，这里为channel数
        
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)#(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1  = nn.Linear(64*40*40+32*7*7, 1024)#64*28*28+32*7*7
        
        self.fc2 = nn.Linear(1024, 1)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32)
            )
            
    def forward(self, x, signal):
        # print(x.size())#x总共元素个数
        # print(signal.size())
        # print(signal.size(0).shape)
        batch_size = signal.size(0)  # 行数532
        x = x.view(batch_size, 1, 40,40)#1, 28,28
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x.view(batch_size, 64*40*40)#64*28*28
        #print(signal.size())
        signal = signal.view(batch_size,20,7,7)
        signal = self.conv3(signal)
        #print(signal.size())
        signal = signal.view(batch_size,32*7*7)
        signal = F.relu(signal)
        
        x = torch.cat([x,signal], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
        
        
        

def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model



class ModelG(nn.Module):

    def __init__(self,in_dim=1,out_dim=1):
        super(ModelG,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating U-Net------\n")

        self.unit_gen = 1
        '''
        self.generation = nn.Sequential(
            # 4*5*5 -> 16*10*10 -> 16*20*20
            nn.Linear(10*7*7,40*40),#64*64
            nn.Sigmoid()

        )  #ding 1101#in_channal,out_channel,kernel_size,stride=1步长,padding=0,dilation=1kernal点间距,groups=1卷积核个数,bias=true
        '''
        self.generation = nn.Sequential(
            # 4*5*5 -> 16*10*10 -> 16*20*20
            nn.ConvTranspose2d(20, 16 * self.unit_gen, kernel_size=8, stride=2, padding=1, bias=False),#kernel_size=4#4*4=28*28,12*16=56*56，12*12=52*52,8*8=40*40,10*10=46*46
            nn.BatchNorm2d(16 * self.unit_gen),
            nn.ReLU(inplace = True),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 20->40->80
            nn.ConvTranspose2d(16 * self.unit_gen, 1, kernel_size=8, stride=2, padding=1, bias=False),#kernel_size=4
            nn.BatchNorm2d(1 * self.unit_gen),
            nn.ReLU(inplace = True),
        )

        self.down_1 = conv_block_2(1,16*2*2,act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(16*2*2,32*2*2,act_fn)
        self.pool_2 = maxpool()

        self.bridge = conv_block_2(32*2*2,64*2*2,act_fn)

        self.trans_1 = conv_trans_block(64*2*2,32*2*2,act_fn)
        self.up_1 = conv_block_2(64*2*2,32*2*2,act_fn)
        self.trans_2 = conv_trans_block(32*2*2,16*2*2,act_fn)
        self.up_2 = conv_block_2(32*2*2,16*2*2,act_fn)

        self.out = nn.Conv2d(16*2*2,self.out_dim,3,1,1)
        self.sig = nn.Sequential(
            nn.Sigmoid()
            )
    def forward(self,x):
        x = x.view(-1,2*10,7,7)
        # print(x.size())#100*20*7*7
        gen = self.generation(x) #ding
        # gen = self.sig(x)  # ding
        # print('gen',gen.size())#100*1*28*28
        gen = gen.view(-1,1,40,40)#-1,1,28,28
        # print('gen', gen.size())  # 100*1*28*28
        #gen = nn.functional.interpolate(gen,size=(28,28), mode='bilinear', align_corners=False),
        #print(gen[0])
        #print(gen[0].size())
        
        down_1 = self.down_1(gen)  #ding
        #down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)


        bridge = self.bridge(pool_2)
        #print("bridge_size", bridge.size())
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1,down_2],dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_1],dim=1)
        up_2 = self.up_2(concat_2)
        
        out = self.out(up_2)
        out = out.view(-1,40*40)#-1,28*28
        out = self.sig(out)
        out  =out.view(-1,1,40,40)#-1,1,28,28
        #print('out',out.size())
        return out

        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=100
    lr = 0.003
    epochs=200
    nz=100                       
    save_every=10
    model_save_dir = 'D:\\Program Files\\JetBrains\\PyCharm Community Edition 2018.3.4\\model1'
    print_every=50   
    criterion=nn.BCELoss()

    train_loader,test_signal,test_GT = loadData(batch_size)
    # with torch.no_grad():
    model_d = ModelD().to(device)
    model_g = ModelG().to(device)

    optim_d = optim.SGD(model_d.parameters(), lr=lr)
    optim_g = optim.SGD(model_g.parameters(), lr=lr)

    

    for epoch_idx in range(epochs):
        print('epoch  : ',epoch_idx)
        d_loss = 0.0
        g_loss = 0.0
        
        for i, (real_signal, real_GT) in enumerate(train_loader):    
            batch_s  = real_signal.shape[0] 
            real_signal = real_signal.to(device)
            real_GT = real_GT.to(device)
            
            real_point = torch.from_numpy(np.ones([batch_s,1])).float().to(device)
            fake_point = torch.from_numpy(np.zeros([batch_s,1])).float().to(device)
            noise_z = Variable(torch.FloatTensor(batch_s,100).normal_(0,1)).to(device)
            noise_signal = Variable(torch.FloatTensor(batch_s,2,10,7,7).normal_(0,1)).to(device)

            #train the D
            # with torch.no_grad():
            output = model_d(real_GT, real_signal)#前向传播
            optim_d.zero_grad()#梯度清零（在反向传播之前即可）
            loss_D_real = criterion(output, real_point)#损失函数
            loss_D_real.backward()#反向传播
            realD_mean = output.data.cpu().mean()
            
            g_out = model_g(real_signal)
            # print(g_out.shape)
            # print(real_signal.shape)
            # print(noise_signal.shape)
            output = model_d(g_out,noise_signal)#where?
            loss_D_fake1 = criterion(output, fake_point)
            loss_D_fake1.backward()
            
            g_out = model_g(real_signal)
            output = model_d(g_out,real_signal)
            loss_D_fake2 = criterion(output, fake_point)
            loss_D_fake2.backward()
            '''
            g_out = model_g(noise_z,noise_label)
            output = model_d(real_image,noise_label)
            loss_D_fake3 = criterion(output, fake_point)
            loss_D_fake3.backward()
            '''
            
            fakeD_mean = output.data.cpu().mean()

            loss_D = loss_D_real + loss_D_fake1 + loss_D_fake2
            optim_d.step()

            # train the G
            for k in range(1):
                g_out = model_g(real_signal)
                output = model_d(g_out,real_signal)
                loss_G = criterion(output,real_point)
                optim_g.zero_grad()
                loss_G.backward()
                optim_g.step()

            # d_loss += loss_D
            d_loss += float(loss_D)
            # g_loss += loss_G
            g_loss += float(loss_G)
            if i % print_every == 0:
                print(
                "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                    format(epoch_idx, i, len(train_loader), fakeD_mean,
                        realD_mean))
        if (epoch_idx+1) % 5 == 0:  
            idx = [123,245,345,445,500,600,700,800,900,950]       
            for j in idx:
                #print()
                #print(signal.size())
                sample_z =  torch.FloatTensor(1,100).normal_(0,1).to(device)
                oriImg = test_GT[j].numpy()
                sample_y = test_signal[j].to(device)
                image = model_g(sample_y)
                image =image.to('cpu')
                image = image.detach().numpy()     
                #print(oriImg)
                #print(f'Iteration {j}. G_loss {g_loss}. D_loss {d_loss}')
                
                image = image.reshape(40,40)#28,28
                oriImg = oriImg.reshape(40,40)#28,28
                image = np.concatenate((image,oriImg),axis=1)
                #save images
                image = plt.imshow(image, cmap = 'Greys_r')
                plt.savefig(r'D:\Program Files\JetBrains\PyCharm Community Edition 2018.3.4\CNN-1\{}_{}.png'.format(epoch_idx,j))
                
            print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
                d_loss, g_loss))
        
        '''
        if epoch_idx >= 50000:
            torch.save(model_d.state_dict(),'{}/model_d_epoch_{}.pkl'.format(
                            model_save_dir, epoch_idx))
            torch.save(model_g.state_dict(),'{}/model_g_epoch_{}.pkl'.format(
                            model_save_dir, epoch_idx))
        '''
