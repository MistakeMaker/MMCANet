import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class SimplifiedAttention(nn.Module):
    def __init__(self, d_model, h,dropout=.1):
        super(SimplifiedAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)


    def forward(self, queries, keys, values, x):
        
        b, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = queries.view(b, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b, h, nq, d_k)
        k = keys.view(b, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b, h, d_k, nk)
        v = values.view(b, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b, h, nk, d_v)

        attention = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b, h, nq, nk)

        attention = torch.softmax(attention, -1)
        attention=self.dropout(attention)

        out = torch.matmul(attention, v)
        out= out.permute(0, 2, 1, 3).contiguous().view(b, nq, self.h * self.d_v) + x# (b, nq, h*d_v)
        out =  out + self.fc(out)
        return out


class MWPT(nn.Module):                                                                                                                                                     ##通过nn.Module来自定义WPT层
    def __init__(self, fil, decom_lvl, flag_kernel):                                                                                                                                          
        super().__init__()                                                                                                                                           
        self.ks = 8                                                 
        self.fil_len = fil.shape[2]                                                                                  
        self.decom_lvl = decom_lvl                                                                                                                                                
        self.bands = 2 ** decom_lvl                                       
        self.mat = self._transform_matrix().cuda()  
        
        pl = (self.ks - self.fil_len) // 2
        self.kernel = nn.Parameter(F.pad(fil, pad=(pl,pl)), requires_grad=flag_kernel)                                                                                                             
                                                                                                                  

    def _transform_matrix(self):                                                                                                                                               
        elems = [(i, (-1)**i) for i in range(self.ks-1, -1, -1)]                                                                                                       
        mat = torch.zeros(self.ks, self.ks)
        for i, (idx,e) in enumerate(elems):
            mat[i,idx] = e
        mat = mat * (-1)**(self.fil_len // 2)
        return mat                                                                                                                               

    def _idx_from_lvl(self, l):                                                                  
        assert l <= self.decom_lvl                                             
        step = self.bands // 2 ** (l - 1)                                                    
        start = self.bands // 2 ** l
        low = list(range(0, self.bands, step))                                               
        high = list(range(start, self.bands + 1, step))
        return low, high

    def _deci2gray(self, i): 
        return i ^ (i >> 1)                                                                    

    def forward(self, x):
        freqs = [0] * self.bands                                              
        freqs[0] = x                                                           
        freqs = self.decomposite(freqs)
        freqs[0].pop(1)
        freqs[0].pop(1)
        freqs[0].pop(1)
        freqs[0].pop(2)
        freqs[0].pop(2)
        freqs[0].pop(2)
        freqs[1].pop(1)  
        freqs[1].pop(2)    
        freqs[1].pop(3) 
        freqs[1].pop(4)                              
        level1 = [freqs[0][self._deci2gray(i)] for i in range(len(freqs[0]))]
        level2 = [freqs[1][self._deci2gray(i)] for i in range(len(freqs[1]))]
        level3 = [freqs[2][self._deci2gray(i)] for i in range(len(freqs[2]))]
        return torch.stack(level1, dim=2),torch.stack(level2, dim=2),torch.stack(level3, dim=2)                         
    
    # 小波包的分解
    def decomposite(self, freqs):
        all_fre=[[0] * 8 for i in range(3)]
        for l in range(1, self.decom_lvl+1):                                   
            i1s, i2s = self._idx_from_lvl(l) 
            # print(i1s)                                                                                                      
            for i1, i2 in zip(i1s, i2s):                                                                                                     
                freqs[i2] = F.conv1d(F.pad(freqs[i1], pad=(self.ks // 2 - 1, self.ks // 2 - 1), mode='circular'), self.kernel @ self.mat, stride=2)  
                freqs[i1] = F.conv1d(F.pad(freqs[i1], pad=(self.ks // 2 - 1, self.ks // 2 - 1), mode='circular'), self.kernel, stride=2) 
                # print(l-1)
                all_fre[l-1][i2]=freqs[i2]
                all_fre[l-1][i1]=freqs[i1]
        return all_fre
# -----------------------------------------------------------------------------
#------------------------------------------------------------------------
m = 4
def backbone(in_chan, out_chan):
    return nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1,bias=False), 
                          nn.BatchNorm2d(out_chan), 
                          nn.ReLU()) 
def conv(in_chan, out_chan):
    return nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1,bias=False), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True)) 

def backbone2(sensor_num, m):
    return nn.Sequential(conv(sensor_num, m), nn.MaxPool2d((1, 2)), 
                          conv(m, 2*m), nn.MaxPool2d((1, 2)),
                          conv(2*m, 4*m), nn.MaxPool2d((1, 2)),
                          conv(4*m, 8*m), nn.MaxPool2d((1, 2)),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten()) 

def cnn_feature():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()) 

class WPTCNN(nn.Module):

    def __init__(self, fil, decom_lvl, sensor_num, output_num, flag_kernel=True,mode=8,fil_fusion=True,fil_num=2):                
        super().__init__()
        self.wpt1 = MWPT(fil[0], 3, flag_kernel) 
        self.fil_fusion = fil_fusion
        self.fil_num = fil_num
        if self.fil_fusion == True and self.fil_num == 2:
            self.wpt2 = MWPT(fil[1], 3,False) 
        elif self.fil_fusion == True and self.fil_num == 3:
            self.wpt2 = MWPT(fil[1], 3,False)
            self.wpt3 = MWPT(fil[2], 3,False)

        self.mode = mode
        self.average_weight = nn.Parameter(torch.ones((1, 1, 1, 30))/30,requires_grad=False)
        self.average_bias = nn.Parameter(torch.zeros(1),requires_grad=False)
        
        
        self.self_attention1 = SimplifiedAttention(d_model=512, h=8)
        self.self_attention2 = SimplifiedAttention(d_model=256, h=8)
        self.self_attention3 = SimplifiedAttention(d_model=128, h=8)
        self.self_attention11 = SimplifiedAttention(d_model=512, h=8)
        self.self_attention22 = SimplifiedAttention(d_model=256, h=8)
        self.self_attention33 = SimplifiedAttention(d_model=128, h=8)

        self.layers1 = backbone2(2, m)
        self.layers2 = backbone2(2, m)
        self.layers3 = backbone2(2, m)

     
        self.lin1 = nn.Linear(3*8*m, 8*m)
        self.lin2 = nn.Linear(8*m, output_num)

            
            

        
    def forward(self, x):   
        f1,f2,f3 = self.wpt1(x[:,:,:])
        f4,f5,f6 = self.wpt2(x[:,:,:])       

        f11 = self.self_attention1(f1.squeeze(1),f1.squeeze(1),f1.squeeze(1),f1.squeeze(1)).reshape(-1,1,2,512)

        f22 = self.self_attention2(f2.squeeze(1),f2.squeeze(1),f2.squeeze(1),f2.squeeze(1)).reshape(-1,1,4,256)

        f33 = self.self_attention3(f3.squeeze(1),f3.squeeze(1),f3.squeeze(1),f3.squeeze(1)).reshape(-1,1,8,128)
        
        f14 = self.self_attention11(f4.squeeze(1),f1.squeeze(1),f1.squeeze(1),f1.squeeze(1)).reshape(-1,1,2,512)

        f25 = self.self_attention22(f5.squeeze(1),f2.squeeze(1),f2.squeeze(1),f2.squeeze(1)).reshape(-1,1,4,256)

        f36 = self.self_attention33(f6.squeeze(1),f3.squeeze(1),f3.squeeze(1),f3.squeeze(1)).reshape(-1,1,8,128)
        
        f1 = torch.cat((f11,f14),dim=1)
        f2 = torch.cat((f22,f25),dim=1)
        f3 = torch.cat((f33,f36),dim=1)

        f1 = self.layers1(f1) 
        f2 = self.layers2(f2)
        f3 = self.layers3(f3)

        f = torch.cat((f1,f2,f3),dim=1)
        outputs = self.lin2(self.lin1(f))

                
        return outputs 
    
    
