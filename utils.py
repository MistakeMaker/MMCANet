import torch
import torch.nn as nn
import pandas as pd
import time
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
# import layers

folder_path = r"E:\unstartprediction\ICR"
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def crossEntropyLoss(outputs, labels):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels.long())                                            
    return loss

def train_func(dataloader, net, optimizer):
    ''' 训练模型 '''
    net.train()
    k, running_loss = 0, 0.0
    for inputs, labels in dataloader:
        inputs, labels, net = inputs[:,:,:].to(device), labels.to(device), net.to(device)
            
        optimizer.zero_grad() 
        outputs = net(inputs)
        #print(outputs)
        #print(labels)
        loss = crossEntropyLoss(outputs, labels) 
        loss.backward() 
        optimizer.step()   
        
        running_loss += loss.cpu().item()
        k += 1
        
    running_loss = running_loss / k
    return net, running_loss

def test_func(dataloader, net):
    net.eval()
    predict=[]
    origin_label=[]
    with torch.no_grad():  
        correct, total = 0, 0                                                                                 
        for inputs, labels in dataloader:                                                                                      
            inputs = inputs[:,:,:].to(device)  
            start=time.time()                              
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)
            end=time.time() 
            # print((end-start)*1000)
            correct += (predicted.cpu()==labels).sum().item()
            total += labels.shape[0]
            predict.append(np.array(predicted.cpu()))
            origin_label.append(np.array(labels.cpu()))

        acc = correct / total
    return acc,np.concatenate(predict),np.concatenate(origin_label)


    
def train_and_val(train_dataloader, val_dataloader, net, num_epochs):
    ''' 训练、验证与测试 '''
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999)) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_loss = []
    for epoch in range(num_epochs): 
        if epoch%50 ==0:
            print("epoch:"+str(epoch+1))
        net, loss = train_func(train_dataloader, net, optimizer) 

        train_loss.append(loss)
        scheduler.step()

    plt.figure()
    plt.plot(train_loss)
    return net,train_loss


def train_and_test(train_dataloader, val_dataloader, model):
    val_acc = test_func(val_dataloader, model)
    return val_acc