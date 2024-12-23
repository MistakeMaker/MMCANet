import torch
import torch.nn as nn
import pandas as pd
import time
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import layers
import copy
import utils
folder_path = r"E:\unstartprediction\ICR"
device = 'cuda' if torch.cuda.is_available() else 'cpu'



FILTERS = {   'sym2':torch.tensor([[[0.482962913144690,0.836516303737469,0.224143868041857,-0.129409522550921]]]),
              'db1': torch.tensor([[[0.707106781186548,0.707106781186548]]]),
              'db2': torch.tensor([[[0,0,0.482962913144690,0.836516303737469,0.224143868041857,-0.129409522550921,0,0]]]),
              'db3': torch.tensor([[[0,0.332670552950957,0.806891509313339,0.459877502119331,-0.135011020010391,-0.0854412738822415,0.0352262918821007,0]]]),
              'db4': torch.tensor([[[0.230377813308855,0.714846570552542,0.630880767929590,-0.0279837694169839,-0.187034811718881,0.0308413818359870,0.0328830116669829,-0.0105974017849973]]]),
              'db5': torch.tensor([[[0.160102397974125,0.603829269797473,0.724308528438574,0.138428145901103,-0.242294887066190,-0.032244869585030,0.077571493840065,-0.006241490213012,-0.012580751999016,0.003335725285002]]]),
              'db6': torch.tensor([[[0.111540743350080,0.494623890398385,0.751133908021578,0.315250351709243,-0.226264693965169,-0.129766867567096,0.0975016055870794,0.0275228655300163,-0.0315820393180312,0.000553842200993802,0.00477725751101065,-0.00107730108499558]]]),
              'db7': torch.tensor([[[0.0778520540850624,0.396539319482306,0.729132090846555,0.469782287405359,-0.143906003929106,-0.224036184994166,0.0713092192670500,0.0806126091510659,-0.0380299369350346,-0.0165745416310156,0.0125509985560138,0.000429577973004703,-0.00180164070399983,0.000353713800001040]]]),
              'db8': torch.tensor([[[0.0544158422430816,0.312871590914466,0.675630736298013,0.585354683654869,-0.0158291052560239,-0.284015542962428,0.000472484573997973,0.128747426620186,-0.0173693010020221,-0.0440882539310647,0.0139810279170155,0.00874609404701566,-0.00487035299301066,-0.000391740372995977,0.000675449405998557,-0.000117476784002282]]]),
          }

train_ind_list={}
train_ind_list['a']=["1.21","1.42", "1.67","1.91", "2.04"]
train_ind_list['b']=["1.21","1.30", "1.79","1.91", "2.04"]


val_ind_list = ["1.54"]      

test_ind_list={}
test_ind_list['a']=["1.30","1.79"]   
test_ind_list['b']=["1.42", "1.67"]  


unstart_time={'1.21':113474,'1.30':133748,'1.42':145004,'1.67':175934,'1.79':193922,'1.91':216468,'2.04':225684}
             

training_loss = []


#cross attention
fil=[0,0,0,0]
fil[0]=FILTERS['db1']
fil[1]=FILTERS['db2']
fil[2]=FILTERS['db3']
fil[3]=FILTERS['db4']

decom_lvl = 3
output_num = 2


num_epochs = 200


train_sch=['a','b']


basic_net = layers.WPTCNN(fil, decom_lvl, 1, output_num)



Sensor = ['4','5','7','8','9','10']
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
for i in train_sch:
    print(i)
    all_predict=[]
    all_label=[]
    for sensor in Sensor:
        print(sensor)
        net = copy.deepcopy(basic_net)
        train_dataloader = read_data(train_ind_list[i],sensor,changepoint_wca[sensor])
    
        val_dataloader = read_data_val(val_ind_list,sensor,changepoint_wca[sensor],batch_size=1024)  ##不需要更改
    
        model,_= utils.train_and_val(train_dataloader, val_dataloader, net, num_epochs)

        val_acc,_,_ = utils.test_func(val_dataloader, model)

        # print((end-start)*1000/2158)
        
        test_dataloader = utils.read_data_val(test_ind_list[i],sensor,changepoint_wca[sensor],batch_size=1024)
        test_acc,predict,origin_label = utils.test_func(test_dataloader, model)
        all_predict.append(predict)
        all_label.append(origin_label)
        print(round(test_acc,4))

    
    predict_label=np.concatenate(all_predict)
    true_label=np.concatenate(all_label)
        


    cm = confusion_matrix(true_label, predict_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap=plt.cm.Blues, values_format='d') 
    plt.show()
    correct_predictions = np.sum(true_label == predict_label)
    total_samples = len(true_label)
    accuracy = correct_predictions / total_samples
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")

